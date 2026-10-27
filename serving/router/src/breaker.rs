//! Per-backend circuit breaker.
//!
//! Closed: traffic flows, consecutive failures are counted. After `threshold` in a row it opens. Open: no traffic for `cooldown`.
//! After the cooldown exactly ONE request is let through (half-open); if it succeeds the breaker closes, if it fails it opens again.
//! Time is passed in, never read, so the state machine is tested deterministically.

use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Closed,
    HalfOpen,
    Open,
}

impl State {
    pub fn as_metric(self) -> u8 {
        match self {
            State::Closed => 0,
            State::HalfOpen => 1,
            State::Open => 2,
        }
    }
}

#[derive(Debug)]
struct Inner {
    state: State,
    failures: u32,
    opened_at: Option<Instant>,
    probe_in_flight: bool,
}

#[derive(Debug)]
pub struct Breaker {
    inner: Mutex<Inner>,
    threshold: u32,
    cooldown: Duration,
}

impl Breaker {
    pub fn new(threshold: u32, cooldown: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner { state: State::Closed, failures: 0, opened_at: None, probe_in_flight: false }),
            threshold: threshold.max(1),
            cooldown,
        }
    }

    /// Would a request be allowed right now? No side effects, so backend selection can look at many backends.
    pub fn peek(&self, now: Instant) -> bool {
        let g = self.inner.lock().unwrap();
        match g.state {
            State::Closed => true,
            State::Open => g.opened_at.map_or(true, |t| now.duration_since(t) >= self.cooldown),
            State::HalfOpen => !g.probe_in_flight,
        }
    }

    /// Reserve the right to send a request. In the half-open state this takes the single probe slot.
    pub fn acquire(&self, now: Instant) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.state {
            State::Closed => true,
            State::Open => {
                if g.opened_at.map_or(true, |t| now.duration_since(t) >= self.cooldown) {
                    g.state = State::HalfOpen;
                    g.probe_in_flight = true;
                    true
                } else {
                    false
                }
            }
            State::HalfOpen => {
                if g.probe_in_flight {
                    false
                } else {
                    g.probe_in_flight = true;
                    true
                }
            }
        }
    }

    pub fn on_success(&self) {
        let mut g = self.inner.lock().unwrap();
        g.state = State::Closed;
        g.failures = 0;
        g.opened_at = None;
        g.probe_in_flight = false;
    }

    pub fn on_failure(&self, now: Instant) {
        let mut g = self.inner.lock().unwrap();
        match g.state {
            State::HalfOpen => {
                g.state = State::Open;
                g.opened_at = Some(now);
                g.probe_in_flight = false;
            }
            State::Closed => {
                g.failures += 1;
                if g.failures >= self.threshold {
                    g.state = State::Open;
                    g.opened_at = Some(now);
                }
            }
            State::Open => {}
        }
    }

    /// A reserved probe ended without an outcome (the client went away): free the slot so the breaker cannot get stuck half-open.
    pub fn release_probe(&self) {
        let mut g = self.inner.lock().unwrap();
        if g.state == State::HalfOpen {
            g.probe_in_flight = false;
        }
    }

    pub fn state(&self) -> State {
        self.inner.lock().unwrap().state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COOL: Duration = Duration::from_millis(1000);

    #[test]
    fn opens_after_threshold_consecutive_failures_only() {
        let b = Breaker::new(3, COOL);
        let t0 = Instant::now();
        b.on_failure(t0);
        b.on_failure(t0);
        b.on_success(); // a success resets the count
        b.on_failure(t0);
        b.on_failure(t0);
        assert_eq!(b.state(), State::Closed, "failures were not consecutive");
        b.on_failure(t0);
        assert_eq!(b.state(), State::Open);
    }

    #[test]
    fn open_blocks_until_cooldown_then_lets_exactly_one_probe_through() {
        let b = Breaker::new(1, COOL);
        let t0 = Instant::now();
        b.on_failure(t0);
        assert!(!b.peek(t0 + Duration::from_millis(999)));
        assert!(!b.acquire(t0 + Duration::from_millis(999)));
        let later = t0 + COOL;
        assert!(b.peek(later));
        assert!(b.acquire(later), "first request after the cooldown is the probe");
        assert_eq!(b.state(), State::HalfOpen);
        assert!(!b.peek(later) && !b.acquire(later), "only one probe at a time");
    }

    #[test]
    fn successful_probe_closes_failed_probe_reopens_with_a_fresh_cooldown() {
        let b = Breaker::new(1, COOL);
        let t0 = Instant::now();
        b.on_failure(t0);
        assert!(b.acquire(t0 + COOL));
        b.on_success();
        assert_eq!(b.state(), State::Closed);
        assert!(b.acquire(t0 + COOL));

        let b = Breaker::new(1, COOL);
        b.on_failure(t0);
        let t1 = t0 + COOL;
        assert!(b.acquire(t1));
        b.on_failure(t1);
        assert_eq!(b.state(), State::Open);
        assert!(!b.peek(t1 + Duration::from_millis(500)), "cooldown restarts from the failed probe");
        assert!(b.peek(t1 + COOL));
    }

    #[test]
    fn abandoned_probe_does_not_wedge_the_breaker() {
        let b = Breaker::new(1, COOL);
        let t0 = Instant::now();
        b.on_failure(t0);
        assert!(b.acquire(t0 + COOL));
        assert!(!b.peek(t0 + COOL));
        b.release_probe();
        assert!(b.peek(t0 + COOL), "slot is free again after release");
    }

    #[test]
    fn peek_has_no_side_effects() {
        let b = Breaker::new(1, COOL);
        let t0 = Instant::now();
        b.on_failure(t0);
        for _ in 0..5 {
            assert!(b.peek(t0 + COOL));
        }
        assert_eq!(b.state(), State::Open, "peek must not move the state to half-open");
    }
}
