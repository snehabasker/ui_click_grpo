//! Choosing a replica.
//!
//! `Prefix` is consistent hashing with bounded load (Mirrokni, Thorup, Zadimoghaddam): walk the ring clockwise from the key's owner and
//! take the first eligible replica whose in-flight count is below `ceil(load_factor * (total + 1) / eligible)`. A busy owner therefore
//! spills its keys to the next replica instead of queueing, and if every candidate is at the cap we fall back to least-loaded.
//! `RoundRobin` and `LeastLoaded` exist so the benchmark compares like with like.

use crate::backend::Backend;
use crate::config::PolicyKind;
use crate::ring::Ring;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// The first eligible replica on the ring (the key's owner, or its successor when the owner is down).
    Affinity,
    /// The owner was above the load cap, so a later replica on the ring took the request.
    Spill,
    LeastLoaded,
    RoundRobin,
}

pub struct Selector {
    kind: PolicyKind,
    ring: Ring,
    rr: AtomicUsize,
    load_factor: f64,
}

fn eligible(b: &Backend, now: Instant) -> bool {
    b.healthy.load(Relaxed) && b.breaker.peek(now)
}

fn least_loaded(elig: &[usize], backends: &[Arc<Backend>]) -> usize {
    *elig.iter().min_by_key(|i| backends[**i].in_flight.load(Relaxed)).expect("non-empty")
}

impl Selector {
    pub fn new(kind: PolicyKind, names: &[String], vnodes: usize, load_factor: f64) -> Self {
        Self { kind, ring: Ring::new(names, vnodes), rr: AtomicUsize::new(0), load_factor: load_factor.max(1.0) }
    }

    /// `has_key` is false for requests with no usable prefix (for example GET /v1/models); they just go to the least-loaded replica.
    pub fn choose(
        &self,
        backends: &[Arc<Backend>],
        key: u64,
        has_key: bool,
        exclude: &[usize],
        now: Instant,
    ) -> Option<(usize, Decision)> {
        let elig: Vec<usize> = (0..backends.len()).filter(|i| !exclude.contains(i) && eligible(&backends[*i], now)).collect();
        if elig.is_empty() {
            return None;
        }
        match self.kind {
            PolicyKind::RoundRobin => {
                let n = self.rr.fetch_add(1, Relaxed);
                Some((elig[n % elig.len()], Decision::RoundRobin))
            }
            PolicyKind::LeastLoaded => Some((least_loaded(&elig, backends), Decision::LeastLoaded)),
            PolicyKind::Prefix => {
                if !has_key {
                    return Some((least_loaded(&elig, backends), Decision::LeastLoaded));
                }
                let total: usize = elig.iter().map(|i| backends[*i].in_flight.load(Relaxed)).sum();
                let cap = ((self.load_factor * (total as f64 + 1.0) / elig.len() as f64).ceil() as usize).max(1);
                let mut first_eligible = None;
                for idx in self.ring.candidates(key) {
                    if !elig.contains(&idx) {
                        continue;
                    }
                    first_eligible.get_or_insert(idx);
                    if backends[idx].in_flight.load(Relaxed) < cap {
                        let d = if Some(idx) == first_eligible { Decision::Affinity } else { Decision::Spill };
                        return Some((idx, d));
                    }
                }
                Some((least_loaded(&elig, backends), Decision::LeastLoaded))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::hash64;
    use std::time::Duration;

    fn backends(n: usize) -> Vec<Arc<Backend>> {
        (0..n).map(|i| Arc::new(Backend::new(&format!("http://r{i}:8000"), 3, Duration::from_secs(1)))).collect()
    }
    fn names(bs: &[Arc<Backend>]) -> Vec<String> {
        bs.iter().map(|b| b.base.clone()).collect()
    }
    fn set_load(bs: &[Arc<Backend>], loads: &[usize]) {
        for (b, l) in bs.iter().zip(loads) {
            b.in_flight.store(*l, Relaxed);
        }
    }

    #[test]
    fn same_key_goes_to_the_same_replica_when_idle() {
        let bs = backends(3);
        let sel = Selector::new(PolicyKind::Prefix, &names(&bs), 160, 1.25);
        let key = hash64(b"system prompt");
        let now = Instant::now();
        let first = sel.choose(&bs, key, true, &[], now).unwrap();
        assert_eq!(first.1, Decision::Affinity);
        for _ in 0..20 {
            assert_eq!(sel.choose(&bs, key, true, &[], now).unwrap().0, first.0);
        }
    }

    #[test]
    fn a_busy_owner_spills_to_the_next_replica_on_the_ring() {
        let bs = backends(3);
        let sel = Selector::new(PolicyKind::Prefix, &names(&bs), 160, 1.25);
        let key = hash64(b"hot prefix");
        let now = Instant::now();
        let owner = sel.choose(&bs, key, true, &[], now).unwrap().0;
        let mut loads = vec![0usize; 3];
        loads[owner] = 12;
        set_load(&bs, &loads);
        let (idx, d) = sel.choose(&bs, key, true, &[], now).unwrap();
        assert_ne!(idx, owner, "owner holds 12 of 12 in-flight, far above the cap");
        assert_eq!(d, Decision::Spill);
    }

    #[test]
    fn the_cap_follows_the_average_so_light_balanced_load_keeps_affinity() {
        let bs = backends(3);
        let sel = Selector::new(PolicyKind::Prefix, &names(&bs), 160, 1.25);
        let key = hash64(b"k");
        let now = Instant::now();
        let owner = sel.choose(&bs, key, true, &[], now).unwrap().0;
        set_load(&bs, &[10, 10, 10]); // cap = ceil(1.25 * 31 / 3) = 13
        let (idx, d) = sel.choose(&bs, key, true, &[], now).unwrap();
        assert_eq!((idx, d), (owner, Decision::Affinity));
    }

    #[test]
    fn unhealthy_owner_hands_its_keys_to_the_successor_and_that_still_counts_as_affinity() {
        let bs = backends(3);
        let sel = Selector::new(PolicyKind::Prefix, &names(&bs), 160, 1.25);
        let key = hash64(b"k2");
        let now = Instant::now();
        let owner = sel.choose(&bs, key, true, &[], now).unwrap().0;
        bs[owner].healthy.store(false, Relaxed);
        let (idx, d) = sel.choose(&bs, key, true, &[], now).unwrap();
        assert_ne!(idx, owner);
        assert_eq!(d, Decision::Affinity);
        bs[owner].healthy.store(true, Relaxed);
        assert_eq!(sel.choose(&bs, key, true, &[], now).unwrap().0, owner, "keys return home when the owner recovers");
    }

    #[test]
    fn open_breaker_and_exclusions_are_respected_and_none_when_nothing_is_left() {
        let bs = backends(2);
        let sel = Selector::new(PolicyKind::Prefix, &names(&bs), 160, 1.25);
        let key = hash64(b"k3");
        let now = Instant::now();
        let owner = sel.choose(&bs, key, true, &[], now).unwrap().0;
        for _ in 0..3 {
            bs[owner].breaker.on_failure(now);
        }
        assert_ne!(sel.choose(&bs, key, true, &[], now).unwrap().0, owner);
        let other = 1 - owner;
        assert!(sel.choose(&bs, key, true, &[other], now).is_none(), "owner's breaker is open and the other is excluded");
    }

    #[test]
    fn round_robin_cycles_and_least_loaded_picks_the_minimum() {
        let bs = backends(3);
        let now = Instant::now();
        let rr = Selector::new(PolicyKind::RoundRobin, &names(&bs), 160, 1.25);
        let picks: Vec<usize> = (0..6).map(|_| rr.choose(&bs, 0, true, &[], now).unwrap().0).collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1, 2]);
        let ll = Selector::new(PolicyKind::LeastLoaded, &names(&bs), 160, 1.25);
        set_load(&bs, &[5, 1, 3]);
        assert_eq!(ll.choose(&bs, 0, true, &[], now).unwrap().0, 1);
    }

    #[test]
    fn requests_without_a_prefix_go_to_the_least_loaded_replica() {
        let bs = backends(3);
        let sel = Selector::new(PolicyKind::Prefix, &names(&bs), 160, 1.25);
        set_load(&bs, &[4, 0, 2]);
        let (idx, d) = sel.choose(&bs, 0, false, &[], Instant::now()).unwrap();
        assert_eq!((idx, d), (1, Decision::LeastLoaded));
    }

    #[test]
    fn bounded_load_keeps_a_single_hot_key_from_overloading_one_replica() {
        // 100 requests for ONE key arrive while nothing completes: without the bound all 100 would pile onto one replica.
        let bs = backends(4);
        let sel = Selector::new(PolicyKind::Prefix, &names(&bs), 160, 1.25);
        let key = hash64(b"one very popular prefix");
        let now = Instant::now();
        for _ in 0..100 {
            let (idx, _) = sel.choose(&bs, key, true, &[], now).unwrap();
            bs[idx].in_flight.fetch_add(1, Relaxed);
        }
        let loads: Vec<usize> = bs.iter().map(|b| b.in_flight.load(Relaxed)).collect();
        let max = *loads.iter().max().unwrap();
        assert_eq!(loads.iter().sum::<usize>(), 100, "no request is dropped");
        // What bounded load guarantees is an UPPER bound: ceil(1.25 * 100 / 4) = 32. It does not guarantee a lower bound: a hot key fills
        // the replicas in ring order, so the last one on the ring gets whatever is left (here about 6).
        assert!(max <= 32, "max in-flight {max} of 100 over 4 replicas (ideal 25, cap 32): {loads:?}");
        assert!(loads.iter().filter(|l| **l > 0).count() >= 3, "the hot key must spill beyond one replica: {loads:?}");
    }
}
