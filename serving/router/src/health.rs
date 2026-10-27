//! Active health checks. Each replica is polled on its own task; it flips to unhealthy after `unhealthy_after` consecutive failures and
//! back after `healthy_after` consecutive successes. (Passive failures seen on real requests feed the circuit breaker instead.)

use crate::backend::Backend;
use crate::config::Config;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;

pub fn spawn(backends: &[Arc<Backend>], cfg: &Config) -> Vec<JoinHandle<()>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(cfg.health_timeout_ms))
        .pool_max_idle_per_host(0)
        .build()
        .expect("health client");
    backends
        .iter()
        .map(|b| {
            let b = b.clone();
            let client = client.clone();
            let url = format!("{}{}", b.base, cfg.health_path);
            let every = Duration::from_millis(cfg.health_interval_ms);
            let (down_after, up_after) = (cfg.unhealthy_after.max(1), cfg.healthy_after.max(1));
            tokio::spawn(async move {
                let (mut ok_streak, mut bad_streak) = (0u32, 0u32);
                loop {
                    tokio::time::sleep(every).await;
                    let good = matches!(client.get(&url).send().await, Ok(r) if r.status().is_success());
                    if good {
                        ok_streak += 1;
                        bad_streak = 0;
                        if !b.healthy.load(Relaxed) && ok_streak >= up_after {
                            b.healthy.store(true, Relaxed);
                            b.stats.came_up.fetch_add(1, Relaxed);
                            tracing::info!(backend = %b.base, "replica is healthy again");
                        }
                    } else {
                        bad_streak += 1;
                        ok_streak = 0;
                        if b.healthy.load(Relaxed) && bad_streak >= down_after {
                            b.healthy.store(false, Relaxed);
                            b.stats.went_down.fetch_add(1, Relaxed);
                            tracing::warn!(backend = %b.base, "replica marked unhealthy");
                        }
                    }
                }
            })
        })
        .collect()
}
