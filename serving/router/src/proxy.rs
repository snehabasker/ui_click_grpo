//! The request path.
//!
//! Read the body, derive the prefix key, pick a replica, forward, and stream the answer back. An attempt that fails BEFORE the response
//! starts (connect error, timeout waiting for headers, 5xx) is retried on a different replica with jittered exponential backoff. Once
//! bytes are flowing to the client the router cannot replay them, so a failure after that point is surfaced as a broken stream and counted
//! against the replica; it is never silently retried.

use crate::backend::Backend;
use crate::config::Config;
use crate::metrics::Metrics;
use crate::policy::Selector;
use crate::prefix::{prefix_key, KeySource};
use crate::ring::splitmix64;
use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{header, HeaderName, Method, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::Value;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct App {
    pub cfg: Config,
    pub backends: Vec<Arc<Backend>>,
    pub selector: Selector,
    pub client: reqwest::Client,
    pub metrics: Metrics,
    pub started: Instant,
}

pub fn build_app(cfg: Config) -> (Router, Arc<App>) {
    let backends: Vec<Arc<Backend>> = cfg
        .backends
        .iter()
        .map(|u| Arc::new(Backend::new(u, cfg.breaker_threshold, Duration::from_millis(cfg.breaker_cooldown_ms))))
        .collect();
    let names: Vec<String> = backends.iter().map(|b| b.base.clone()).collect();
    let selector = Selector::new(cfg.policy, &names, cfg.vnodes, cfg.load_factor);
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(cfg.connect_timeout_ms))
        .pool_max_idle_per_host(256)
        .tcp_nodelay(true)
        .build()
        .expect("http client");
    let app = Arc::new(App { cfg, backends, selector, client, metrics: Metrics::default(), started: Instant::now() });
    let router = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/router/health", get(health_handler))
        .fallback(proxy_handler)
        .with_state(app.clone());
    (router, app)
}

async fn metrics_handler(State(app): State<Arc<App>>) -> Response {
    let body = app.metrics.render(&app.backends, app.started.elapsed());
    Response::builder()
        .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(Body::from(body))
        .unwrap()
}

async fn health_handler(State(app): State<Arc<App>>) -> Response {
    let now = Instant::now();
    let mut healthy = 0;
    let items: Vec<String> = app
        .backends
        .iter()
        .map(|b| {
            let h = b.healthy.load(Relaxed);
            healthy += h as usize;
            format!(
                "{{\"url\":\"{}\",\"healthy\":{},\"breaker\":\"{:?}\",\"in_flight\":{},\"accepting\":{}}}",
                b.base,
                h,
                b.breaker.state(),
                b.in_flight.load(Relaxed),
                h && b.breaker.peek(now)
            )
        })
        .collect();
    let status = if healthy > 0 { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(format!("{{\"healthy\":{healthy},\"backends\":[{}]}}", items.join(","))))
        .unwrap()
}

fn error_response(status: StatusCode, msg: &str) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::json!({"error": {"message": msg, "type": "router_error"}}).to_string()))
        .unwrap()
}

/// Headers that must not be copied between the two HTTP hops.
fn hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection" | "keep-alive" | "proxy-authenticate" | "proxy-authorization" | "te" | "trailer" | "transfer-encoding" | "upgrade" | "host" | "content-length"
    )
}

/// Exponential backoff without jitter: base, 2x, 4x, ... capped at `max`.
pub fn backoff_base(attempt: u32, base_ms: u64, max_ms: u64) -> u64 {
    base_ms.saturating_mul(1u64 << attempt.min(16)).min(max_ms)
}

/// The same with +-25% jitter so retries from many requests do not synchronise.
fn backoff(attempt: u32, base_ms: u64, max_ms: u64) -> Duration {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    let r = splitmix64(u64::from(nanos) ^ (u64::from(attempt) << 32)) % 1000; // 0..999
    let factor = 0.75 + (r as f64 / 1000.0) * 0.5;
    Duration::from_millis((backoff_base(attempt, base_ms, max_ms) as f64 * factor) as u64)
}

/// Tracks one attempt on one replica: in-flight accounting and exactly one outcome for the breaker and the counters.
struct Guard {
    app: Arc<App>,
    backend: Arc<Backend>,
    start: Instant,
    finished: bool,
}

impl Guard {
    fn new(app: Arc<App>, backend: Arc<Backend>, start: Instant) -> Self {
        backend.in_flight.fetch_add(1, Relaxed);
        Self { app, backend, start, finished: false }
    }
    fn complete(&mut self) {
        if !self.finished {
            self.finished = true;
            self.backend.breaker.on_success();
            self.backend.stats.ok.fetch_add(1, Relaxed);
            self.app.metrics.total.observe(self.start.elapsed());
        }
    }
    fn fail(&mut self) {
        if !self.finished {
            self.finished = true;
            self.backend.breaker.on_failure(Instant::now());
            self.backend.stats.err.fetch_add(1, Relaxed);
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.backend.in_flight.fetch_sub(1, Relaxed);
        if !self.finished {
            // the client went away mid-response: not the replica's fault, but free a half-open probe slot if we held it
            self.backend.stats.aborted.fetch_add(1, Relaxed);
            self.backend.breaker.release_probe();
        }
    }
}

async fn proxy_handler(State(app): State<Arc<App>>, req: Request) -> Response {
    let started = Instant::now();
    app.metrics.requests.fetch_add(1, Relaxed);
    let (parts, body) = req.into_parts();
    let bytes = match to_bytes(body, app.cfg.max_body_mb * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return error_response(StatusCode::PAYLOAD_TOO_LARGE, "request body unreadable or too large"),
    };
    let json: Option<Value> = if parts.method == Method::POST { serde_json::from_slice(&bytes).ok() } else { None };
    let (key, source) = prefix_key(&parts.headers, json.as_ref(), app.cfg.prefix_messages);
    let has_key = source != KeySource::None;
    let path_and_query = parts.uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| "/".to_string());

    let header_timeout = Duration::from_millis(app.cfg.header_timeout_ms);
    let idle = Duration::from_millis(app.cfg.idle_timeout_ms);
    let mut tried: Vec<usize> = Vec::new();
    let mut attempt: u32 = 0;
    let mut last_error = String::from("no replica available");

    while attempt <= app.cfg.max_retries {
        let now = Instant::now();
        let Some((idx, decision)) = app.selector.choose(&app.backends, key, has_key, &tried, now) else { break };
        let backend = app.backends[idx].clone();
        if !backend.breaker.acquire(now) {
            tried.push(idx); // lost the race for the half-open probe slot; try another replica without spending a retry
            continue;
        }
        app.metrics.count_decision(decision);
        if attempt > 0 {
            backend.stats.retried_to.fetch_add(1, Relaxed);
            app.metrics.retries.fetch_add(1, Relaxed);
        }
        let mut guard = Guard::new(app.clone(), backend.clone(), started);

        let url = format!("{}{}", backend.base, path_and_query);
        let mut rb = app.client.request(parts.method.clone(), &url).body(bytes.clone());
        for (k, v) in parts.headers.iter() {
            if !hop_by_hop(k) {
                rb = rb.header(k, v);
            }
        }

        match tokio::time::timeout(header_timeout, rb.send()).await {
            Ok(Ok(resp)) => {
                let status = resp.status();
                if status.is_server_error() && attempt < app.cfg.max_retries {
                    guard.fail();
                    last_error = format!("replica {} answered {}", backend.base, status);
                    tried.push(idx);
                    tokio::time::sleep(backoff(attempt, app.cfg.backoff_base_ms, app.cfg.backoff_max_ms)).await;
                    attempt += 1;
                    continue;
                }
                app.metrics.ttfb.observe(started.elapsed());
                if status.is_server_error() {
                    guard.fail(); // out of retries: pass the error through, but it still counts against the replica
                } else {
                    backend.breaker.on_success(); // it answered: close a half-open breaker without waiting for the stream to end
                }
                let mut builder = Response::builder().status(status).header("x-router-backend", idx.to_string());
                for (k, v) in resp.headers().iter() {
                    if !hop_by_hop(k) {
                        builder = builder.header(k, v);
                    }
                }
                let stream = async_stream::stream! {
                    let mut guard = guard;
                    let mut up = Box::pin(resp.bytes_stream());
                    loop {
                        match tokio::time::timeout(idle, up.next()).await {
                            Ok(Some(Ok(chunk))) => yield Ok::<Bytes, std::io::Error>(chunk),
                            Ok(Some(Err(e))) => {
                                guard.fail();
                                yield Err(std::io::Error::other(e));
                                break;
                            }
                            Ok(None) => {
                                guard.complete();
                                break;
                            }
                            Err(_) => {
                                guard.fail();
                                yield Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "replica went silent mid-stream"));
                                break;
                            }
                        }
                    }
                };
                return builder.body(Body::from_stream(stream)).unwrap_or_else(|_| error_response(StatusCode::BAD_GATEWAY, "bad upstream response"));
            }
            Ok(Err(e)) => {
                guard.fail();
                last_error = format!("replica {}: {}", backend.base, e);
            }
            Err(_) => {
                guard.fail();
                last_error = format!("replica {} did not answer within {} ms", backend.base, app.cfg.header_timeout_ms);
            }
        }
        tried.push(idx);
        if attempt < app.cfg.max_retries {
            tokio::time::sleep(backoff(attempt, app.cfg.backoff_base_ms, app.cfg.backoff_max_ms)).await;
        }
        attempt += 1;
    }

    app.metrics.no_backend.fetch_add(1, Relaxed);
    tracing::warn!(error = %last_error, "request failed on every eligible replica");
    let status = if tried.is_empty() { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::BAD_GATEWAY };
    error_response(status, &last_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_is_capped() {
        let v: Vec<u64> = (0..8).map(|a| backoff_base(a, 50, 1000)).collect();
        assert_eq!(v, vec![50, 100, 200, 400, 800, 1000, 1000, 1000]);
        assert_eq!(backoff_base(63, 50, 1000), 1000, "huge attempt numbers must not overflow");
    }

    #[test]
    fn jitter_stays_within_25_percent() {
        for a in 0..5 {
            for _ in 0..200 {
                let d = backoff(a, 100, 100_000).as_millis() as f64;
                let b = backoff_base(a, 100, 100_000) as f64;
                assert!(d >= b * 0.74 && d <= b * 1.26, "attempt {a}: {d} ms vs base {b} ms");
            }
        }
    }

    #[test]
    fn hop_by_hop_headers_are_dropped() {
        for h in ["connection", "transfer-encoding", "host", "content-length", "keep-alive"] {
            assert!(hop_by_hop(&HeaderName::from_static(h)), "{h}");
        }
        for h in ["content-type", "authorization", "accept", "x-prefix-key"] {
            assert!(!hop_by_hop(&HeaderName::from_static(h)), "{h}");
        }
    }
}
