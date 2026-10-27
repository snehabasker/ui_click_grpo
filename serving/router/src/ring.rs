//! Consistent-hash ring with virtual nodes.
//!
//! A key maps to a point on a 64-bit ring; the owner is the first virtual node clockwise. `candidates` returns every backend in
//! clockwise order starting at the owner, so callers can skip unhealthy or overloaded ones. Skipping (rather than rebuilding the ring)
//! is what makes the hashing *consistent*: when a backend is unavailable only the keys it owned move, to the next backend on the ring.

/// FNV-1a over the bytes, then a splitmix64 finaliser so nearby inputs spread over the whole ring. Deterministic across runs and
/// platforms (the standard library hasher makes no such promise).
pub fn hash64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    splitmix64(h)
}

pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[derive(Debug, Clone)]
pub struct Ring {
    points: Vec<(u64, usize)>,
    n: usize,
}

impl Ring {
    /// `names` identify the backends (their position is the index returned later); `vnodes` virtual nodes per backend.
    pub fn new(names: &[String], vnodes: usize) -> Self {
        let mut points = Vec::with_capacity(names.len() * vnodes);
        for (i, name) in names.iter().enumerate() {
            for v in 0..vnodes.max(1) {
                points.push((hash64(format!("{name}#{v}").as_bytes()), i));
            }
        }
        points.sort_unstable();
        Self { points, n: names.len() }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Backend indices in clockwise order from the key's position, each exactly once.
    pub fn candidates(&self, key: u64) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.n);
        if self.points.is_empty() {
            return out;
        }
        let start = self.points.partition_point(|p| p.0 < key) % self.points.len();
        let mut seen = vec![false; self.n];
        for off in 0..self.points.len() {
            let idx = self.points[(start + off) % self.points.len()].1;
            if !seen[idx] {
                seen[idx] = true;
                out.push(idx);
                if out.len() == self.n {
                    break;
                }
            }
        }
        out
    }

    /// The owner of a key (first candidate).
    pub fn owner(&self, key: u64) -> Option<usize> {
        self.candidates(key).first().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("http://10.0.0.{i}:8000")).collect()
    }

    #[test]
    fn hash_is_deterministic_and_spreads_nearby_keys() {
        assert_eq!(hash64(b"system prompt A"), hash64(b"system prompt A"));
        assert_ne!(hash64(b"system prompt A"), hash64(b"system prompt B"));
        // a known value pins the algorithm: changing the hash would silently re-shard every deployed key
        assert_eq!(hash64(b""), splitmix64(0xcbf2_9ce4_8422_2325));
    }

    #[test]
    fn candidates_lists_every_backend_once_starting_at_the_owner() {
        let ring = Ring::new(&names(4), 64);
        for key in [0u64, 1, 12345, u64::MAX, hash64(b"x")] {
            let c = ring.candidates(key);
            assert_eq!(c.len(), 4);
            let mut sorted = c.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![0, 1, 2, 3]);
            assert_eq!(ring.owner(key), Some(c[0]));
        }
    }

    #[test]
    fn keys_are_spread_roughly_evenly() {
        let n = 4;
        let ring = Ring::new(&names(n), 160);
        let mut counts = vec![0usize; n];
        let total = 20_000;
        for i in 0..total {
            counts[ring.owner(hash64(format!("key-{i}").as_bytes())).unwrap()] += 1;
        }
        let ideal = total / n;
        for (i, c) in counts.iter().enumerate() {
            let dev = (*c as f64 - ideal as f64).abs() / ideal as f64;
            assert!(dev < 0.20, "backend {i} owns {c} of {total} keys ({:.0}% off ideal)", dev * 100.0);
        }
    }

    #[test]
    fn removing_a_backend_only_moves_the_keys_it_owned() {
        let before = Ring::new(&names(4), 160);
        // "remove" backend 2 by skipping it in the candidate order, exactly what the router does for an unhealthy backend
        let mut moved_wrongly = 0;
        let mut owned_by_removed = 0;
        let total = 10_000;
        for i in 0..total {
            let key = hash64(format!("k{i}").as_bytes());
            let old = before.owner(key).unwrap();
            let new = before.candidates(key).into_iter().find(|b| *b != 2).unwrap();
            if old == 2 {
                owned_by_removed += 1;
            } else if old != new {
                moved_wrongly += 1;
            }
        }
        assert_eq!(moved_wrongly, 0, "keys owned by healthy backends must not move");
        let frac = owned_by_removed as f64 / total as f64;
        assert!((0.15..0.35).contains(&frac), "removed backend owned {:.0}% of keys, expected about 25%", frac * 100.0);
    }

    #[test]
    fn empty_ring_has_no_candidates() {
        let ring = Ring::new(&[], 10);
        assert!(ring.is_empty());
        assert!(ring.candidates(5).is_empty());
        assert_eq!(ring.owner(5), None);
    }
}
