//! Prometheus text exposition for the router. Everything is an atomic, so rendering never blocks the request path.

use crate::backend::Backend;
use crate::policy::Decision;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Duration;

const BOUNDS: [f64; 10] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0];

pub struct Histogram {
    counts: [AtomicU64; BOUNDS.len() + 1],
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self { counts: Default::default(), sum_micros: AtomicU64::new(0), count: AtomicU64::new(0) }
    }
}

impl Histogram {
    pub fn observe(&self, d: Duration) {
        let s = d.as_secs_f64();
        let i = BOUNDS.iter().position(|b| s <= *b).unwrap_or(BOUNDS.len());
        self.counts[i].fetch_add(1, Relaxed);
        self.sum_micros.fetch_add(d.as_micros() as u64, Relaxed);
        self.count.fetch_add(1, Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Relaxed)
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut cum = 0u64;
        for (i, b) in BOUNDS.iter().enumerate() {
            cum += self.counts[i].load(Relaxed);
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {cum}");
        }
        cum += self.counts[BOUNDS.len()].load(Relaxed);
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {cum}");
        let _ = writeln!(out, "{name}_sum {}", self.sum_micros.load(Relaxed) as f64 / 1e6);
        let _ = writeln!(out, "{name}_count {}", self.count.load(Relaxed));
    }
}

#[derive(Default)]
pub struct Metrics {
    decisions: [AtomicU64; 4],
    pub retries: AtomicU64,
    pub no_backend: AtomicU64,
    pub requests: AtomicU64,
    /// Seconds from receiving a request to the replica's response headers.
    pub ttfb: Histogram,
    /// Seconds from receiving a request to the end of the streamed response.
    pub total: Histogram,
}

impl Metrics {
    pub fn count_decision(&self, d: Decision) {
        let i = match d {
            Decision::Affinity => 0,
            Decision::Spill => 1,
            Decision::LeastLoaded => 2,
            Decision::RoundRobin => 3,
        };
        self.decisions[i].fetch_add(1, Relaxed);
    }

    pub fn decision(&self, d: Decision) -> u64 {
        let i = match d {
            Decision::Affinity => 0,
            Decision::Spill => 1,
            Decision::LeastLoaded => 2,
            Decision::RoundRobin => 3,
        };
        self.decisions[i].load(Relaxed)
    }

    pub fn render(&self, backends: &[Arc<Backend>], uptime: Duration) -> String {
        let mut o = String::new();
        let _ = writeln!(o, "# HELP router_uptime_seconds Seconds since start\n# TYPE router_uptime_seconds gauge\nrouter_uptime_seconds {:.1}", uptime.as_secs_f64());
        let _ = writeln!(o, "# HELP router_requests_total Requests received\n# TYPE router_requests_total counter\nrouter_requests_total {}", self.requests.load(Relaxed));
        let _ = writeln!(o, "# HELP router_route_decisions_total How replicas were chosen\n# TYPE router_route_decisions_total counter");
        for (name, d) in [("affinity", Decision::Affinity), ("spill", Decision::Spill), ("least_loaded", Decision::LeastLoaded), ("round_robin", Decision::RoundRobin)] {
            let _ = writeln!(o, "router_route_decisions_total{{decision=\"{name}\"}} {}", self.decision(d));
        }
        let _ = writeln!(o, "# HELP router_retries_total Attempts retried on another replica\n# TYPE router_retries_total counter\nrouter_retries_total {}", self.retries.load(Relaxed));
        let _ = writeln!(o, "# HELP router_no_backend_total Requests rejected because no replica was eligible\n# TYPE router_no_backend_total counter\nrouter_no_backend_total {}", self.no_backend.load(Relaxed));
        let gauge = |o: &mut String, name: &str, help: &str, f: &dyn Fn(&Backend) -> String| {
            let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge");
            for b in backends {
                let _ = writeln!(o, "{name}{{backend=\"{}\"}} {}", b.base, f(b));
            }
        };
        gauge(&mut o, "router_backend_in_flight", "Requests currently being served by the replica", &|b| b.in_flight.load(Relaxed).to_string());
        gauge(&mut o, "router_backend_healthy", "1 if the health checker considers the replica healthy", &|b| (b.healthy.load(Relaxed) as u8).to_string());
        gauge(&mut o, "router_backend_breaker_state", "Circuit breaker: 0 closed, 1 half-open, 2 open", &|b| b.breaker.state().as_metric().to_string());
        let counter = |o: &mut String, name: &str, help: &str, f: &dyn Fn(&Backend) -> u64| {
            let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} counter");
            for b in backends {
                let _ = writeln!(o, "{name}{{backend=\"{}\"}} {}", b.base, f(b));
            }
        };
        counter(&mut o, "router_backend_ok_total", "Responses fully streamed", &|b| b.stats.ok.load(Relaxed));
        counter(&mut o, "router_backend_errors_total", "Failed attempts (connect, timeout, 5xx, broken stream)", &|b| b.stats.err.load(Relaxed));
        counter(&mut o, "router_backend_aborted_total", "Requests whose client disconnected early", &|b| b.stats.aborted.load(Relaxed));
        counter(&mut o, "router_backend_retried_to_total", "Times chosen as the retry target", &|b| b.stats.retried_to.load(Relaxed));
        counter(&mut o, "router_backend_went_down_total", "Healthy to unhealthy transitions", &|b| b.stats.went_down.load(Relaxed));
        counter(&mut o, "router_backend_came_up_total", "Unhealthy to healthy transitions", &|b| b.stats.came_up.load(Relaxed));
        self.ttfb.render(&mut o, "router_ttfb_seconds", "Time to the replica's response headers");
        self.total.render(&mut o, "router_request_duration_seconds", "Time to the end of the streamed response");
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative_and_sum_matches() {
        let h = Histogram::default();
        h.observe(Duration::from_millis(30));
        h.observe(Duration::from_millis(200));
        h.observe(Duration::from_secs(500));
        let mut out = String::new();
        h.render(&mut out, "x", "help");
        assert!(out.contains("x_bucket{le=\"0.05\"} 1"));
        assert!(out.contains("x_bucket{le=\"0.25\"} 2"));
        assert!(out.contains("x_bucket{le=\"+Inf\"} 3"));
        assert!(out.contains("x_count 3"));
        assert_eq!(h.count(), 3);
    }

    #[test]
    fn render_lists_every_backend_with_labels() {
        let bs: Vec<Arc<Backend>> = (0..2).map(|i| Arc::new(Backend::new(&format!("http://r{i}:1"), 3, Duration::from_secs(1)))).collect();
        let m = Metrics::default();
        m.count_decision(Decision::Affinity);
        m.count_decision(Decision::Affinity);
        m.count_decision(Decision::Spill);
        let out = m.render(&bs, Duration::from_secs(5));
        assert!(out.contains("router_route_decisions_total{decision=\"affinity\"} 2"));
        assert!(out.contains("router_route_decisions_total{decision=\"spill\"} 1"));
        assert!(out.contains("router_backend_healthy{backend=\"http://r0:1\"} 1"));
        assert!(out.contains("router_backend_breaker_state{backend=\"http://r1:1\"} 0"));
    }
}
