use crate::breaker::Breaker;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::time::Duration;

/// Counters kept per replica (rendered by the metrics endpoint).
#[derive(Default)]
pub struct BackendStats {
    /// Requests that completed (response fully streamed).
    pub ok: AtomicU64,
    /// Attempts that failed: connect error, timeout, 5xx, or a stream that broke mid-way.
    pub err: AtomicU64,
    /// Requests whose client disconnected before the response finished (not the replica's fault).
    pub aborted: AtomicU64,
    /// Times this replica was chosen as the retry target after another one failed.
    pub retried_to: AtomicU64,
    pub went_down: AtomicU64,
    pub came_up: AtomicU64,
}

pub struct Backend {
    /// Base URL without a trailing slash.
    pub base: String,
    pub in_flight: AtomicUsize,
    /// Set by the health checker; selection never picks an unhealthy replica.
    pub healthy: AtomicBool,
    pub breaker: Breaker,
    pub stats: BackendStats,
}

impl Backend {
    pub fn new(base: &str, breaker_threshold: u32, breaker_cooldown: Duration) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            in_flight: AtomicUsize::new(0),
            // optimistic until the first health check says otherwise, so a router started a moment before its replicas still works
            healthy: AtomicBool::new(true),
            breaker: Breaker::new(breaker_threshold, breaker_cooldown),
            stats: BackendStats::default(),
        }
    }
}
