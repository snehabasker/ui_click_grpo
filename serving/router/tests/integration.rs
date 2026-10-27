//! End-to-end tests: a real router in front of mock replicas that speak the OpenAI streaming format.
//! The mocks can fail, hang, be killed and be restarted on the same port, which is what the failure behaviour is judged on.

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use prefix_router::{build_app, health, App, Config, PolicyKind};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

struct MockState {
    name: String,
    fail: AtomicBool,
    healthy: AtomicBool,
    hits: AtomicU64,
    delay_ms: AtomicU64,
}

struct Mock {
    addr: SocketAddr,
    state: Arc<MockState>,
    task: JoinHandle<()>,
}

impl Mock {
    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
    fn kill(&self) {
        self.task.abort();
    }
}

async fn mock_health(State(s): State<Arc<MockState>>) -> StatusCode {
    if s.healthy.load(SeqCst) { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE }
}

async fn mock_chat(State(s): State<Arc<MockState>>, _body: Bytes) -> Response {
    s.hits.fetch_add(1, SeqCst);
    if s.fail.load(SeqCst) {
        return Response::builder().status(StatusCode::SERVICE_UNAVAILABLE).body(Body::from("mock failure")).unwrap();
    }
    let d = s.delay_ms.load(SeqCst);
    if d > 0 {
        tokio::time::sleep(Duration::from_millis(d)).await; // delays the response HEADERS, like a hung replica
    }
    let name = s.name.clone();
    let stream = async_stream::stream! {
        for i in 0..5 {
            tokio::time::sleep(Duration::from_millis(5)).await;
            yield Ok::<Bytes, Infallible>(Bytes::from(format!("data: {{\"backend\":\"{name}\",\"i\":{i}}}\n\n")));
        }
        yield Ok(Bytes::from("data: [DONE]\n\n"));
    };
    Response::builder().header("content-type", "text/event-stream").header("x-mock", s.name.as_str()).body(Body::from_stream(stream)).unwrap()
}

async fn start_mock(name: &str, bind: Option<SocketAddr>) -> Mock {
    let state = Arc::new(MockState {
        name: name.to_string(),
        fail: AtomicBool::new(false),
        healthy: AtomicBool::new(true),
        hits: AtomicU64::new(0),
        delay_ms: AtomicU64::new(0),
    });
    let app = Router::new().route("/health", get(mock_health)).route("/v1/chat/completions", post(mock_chat)).with_state(state.clone());
    let listener = TcpListener::bind(bind.unwrap_or_else(|| "127.0.0.1:0".parse().unwrap())).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Mock { addr, state, task }
}

struct Rig {
    base: String,
    app: Arc<App>,
    client: reqwest::Client,
    _tasks: Vec<JoinHandle<()>>,
}

async fn start_router(mocks: &[&Mock], policy: PolicyKind) -> Rig {
    start_router_with(mocks, policy, &[]).await
}

async fn start_router_with(mocks: &[&Mock], policy: PolicyKind, overrides: &[(&str, &str)]) -> Rig {
    let urls: Vec<String> = mocks.iter().map(|m| m.url()).collect();
    let cfg = Config::for_tests_with(&urls, policy, overrides);
    let (router, app) = build_app(cfg.clone());
    let mut tasks = health::spawn(&app.backends, &cfg);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tasks.push(tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    }));
    Rig { base, app, client: reqwest::Client::new(), _tasks: tasks }
}

struct Answer {
    status: StatusCode,
    body: String,
    backend: Option<usize>,
}

async fn chat(rig: &Rig, system: &str, user: &str) -> Answer {
    let body = serde_json::json!({"model": "m", "stream": true, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]});
    let r = rig.client.post(format!("{}/v1/chat/completions", rig.base)).header("content-type", "application/json").body(body.to_string()).send().await.unwrap();
    let status = r.status();
    let backend = r.headers().get("x-router-backend").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok());
    Answer { status, body: r.text().await.unwrap_or_default(), backend }
}

async fn wait_until(what: &str, secs: u64, mut cond: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(secs) {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("timed out after {secs}s waiting for: {what}");
}

fn metric(text: &str, series: &str) -> f64 {
    text.lines()
        .find(|l| l.starts_with(series) && l[series.len()..].starts_with(' '))
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("series {series} not in metrics output"))
}

async fn metrics(rig: &Rig) -> String {
    rig.client.get(format!("{}/metrics", rig.base)).send().await.unwrap().text().await.unwrap()
}

#[tokio::test]
async fn same_system_prompt_always_lands_on_the_same_replica() {
    let (a, b, c) = (start_mock("a", None).await, start_mock("b", None).await, start_mock("c", None).await);
    let rig = start_router(&[&a, &b, &c], PolicyKind::Prefix).await;
    let mut used = std::collections::HashSet::new();
    for p in 0..12 {
        let system = format!("you are agent number {p}");
        let mut seen = std::collections::HashSet::new();
        for turn in 0..6 {
            let r = chat(&rig, &system, &format!("turn {turn}")).await;
            assert_eq!(r.status, StatusCode::OK);
            seen.insert(r.backend.unwrap());
        }
        assert_eq!(seen.len(), 1, "prefix {p} was spread over replicas {seen:?}");
        used.extend(seen);
    }
    assert!(used.len() >= 2, "12 different prefixes should not all hash to one replica: {used:?}");
}

#[tokio::test]
async fn explicit_prefix_key_header_overrides_the_system_prompt() {
    let (a, b, c) = (start_mock("a", None).await, start_mock("b", None).await, start_mock("c", None).await);
    let rig = start_router(&[&a, &b, &c], PolicyKind::Prefix).await;
    let mut seen = std::collections::HashSet::new();
    for i in 0..9 {
        let body = serde_json::json!({"messages": [{"role": "system", "content": format!("different every time {i}")}]});
        let r = rig.client.post(format!("{}/v1/chat/completions", rig.base)).header("x-prefix-key", "session-7").header("content-type", "application/json").body(body.to_string()).send().await.unwrap();
        seen.insert(r.headers().get("x-router-backend").unwrap().to_str().unwrap().to_string());
    }
    assert_eq!(seen.len(), 1);
}

#[tokio::test]
async fn round_robin_spreads_requests_evenly() {
    let (a, b, c) = (start_mock("a", None).await, start_mock("b", None).await, start_mock("c", None).await);
    let rig = start_router(&[&a, &b, &c], PolicyKind::RoundRobin).await;
    for i in 0..30 {
        assert_eq!(chat(&rig, "same system prompt for everyone", &format!("{i}")).await.status, StatusCode::OK);
    }
    assert_eq!([&a, &b, &c].map(|m| m.state.hits.load(SeqCst)), [10, 10, 10]);
}

#[tokio::test]
async fn streaming_is_passed_through_complete_and_in_order() {
    let a = start_mock("only", None).await;
    let rig = start_router(&[&a], PolicyKind::Prefix).await;
    let r = chat(&rig, "s", "u").await;
    assert_eq!(r.status, StatusCode::OK);
    let lines: Vec<&str> = r.body.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 6, "5 chunks and [DONE]: {lines:?}");
    for (i, l) in lines.iter().take(5).enumerate() {
        assert!(l.contains(&format!("\"i\":{i}")), "chunk {i} out of order: {l}");
    }
    assert_eq!(lines[5], "data: [DONE]");
}

#[tokio::test]
async fn a_killed_replica_causes_no_client_errors_and_is_marked_down() {
    let (a, b) = (start_mock("a", None).await, start_mock("b", None).await);
    let rig = start_router(&[&a, &b], PolicyKind::Prefix).await;
    for i in 0..10 {
        assert_eq!(chat(&rig, &format!("prompt {i}"), "x").await.status, StatusCode::OK);
    }
    a.kill();
    // from the instant the replica is gone, every request must still succeed: attempts that hit the dead replica are retried elsewhere
    let mut errors = 0;
    for i in 0..40 {
        if chat(&rig, &format!("prompt {}", i % 10), "x").await.status != StatusCode::OK {
            errors += 1;
        }
    }
    assert_eq!(errors, 0, "clients saw errors after a replica died");
    wait_until("replica a marked unhealthy", 3, || !rig.app.backends[0].healthy.load(SeqCst)).await;
    let m = metrics(&rig).await;
    assert_eq!(metric(&m, &format!("router_backend_healthy{{backend=\"{}\"}}", a.url())), 0.0);
    assert_eq!(metric(&m, &format!("router_backend_healthy{{backend=\"{}\"}}", b.url())), 1.0);
    assert!(metric(&m, &format!("router_backend_went_down_total{{backend=\"{}\"}}", a.url())) >= 1.0);
}

#[tokio::test]
async fn before_the_health_check_notices_a_dead_replica_requests_are_retried_and_the_breaker_bounds_the_damage() {
    // health checks effectively off (60 s), so the router still believes the dead replica is healthy and requests really do hit it
    let (a, b) = (start_mock("a", None).await, start_mock("b", None).await);
    let rig = start_router_with(&[&a, &b], PolicyKind::RoundRobin, &[("--health-interval-ms", "60000")]).await;
    a.kill();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let t0 = Instant::now();
    for i in 0..20 {
        assert_eq!(chat(&rig, "s", &format!("{i}")).await.status, StatusCode::OK, "request {i} must survive the dead replica");
    }
    let elapsed_ms = t0.elapsed().as_millis() as f64;
    let m = metrics(&rig).await;
    assert!(metric(&m, "router_retries_total") >= 3.0, "{m}");
    let a_errors = metric(&m, &format!("router_backend_errors_total{{backend=\"{}\"}}", a.url()));
    // The breaker opens after 3 consecutive failures, then lets exactly one probe through per cooldown (250 ms here), and a probe
    // against a dead replica fails again. So failures are bounded by threshold + cooldowns elapsed (+1 for the cooldown in progress),
    // far below the 10 it would take with no breaker: round-robin sends half of the 20 requests to the dead replica.
    let bound = 3.0 + (elapsed_ms / 250.0).floor() + 1.0;
    assert!(a_errors >= 3.0 && a_errors <= bound, "dead replica saw {a_errors} failed attempts in {elapsed_ms} ms, bound {bound}");
    assert!(a_errors < 10.0, "a breaker must beat the no-breaker count of about 10, saw {a_errors}");
    assert_eq!(metric(&m, &format!("router_backend_breaker_state{{backend=\"{}\"}}", a.url())), 2.0);
}

#[tokio::test]
async fn a_restarted_replica_receives_its_keys_again() {
    let (a, b) = (start_mock("a", None).await, start_mock("b", None).await);
    let addr_a = a.addr;
    let rig = start_router(&[&a, &b], PolicyKind::Prefix).await;
    let prompts: Vec<String> = (0..16).map(|i| format!("prompt {i}")).collect();
    let mut owner_a = Vec::new();
    for p in &prompts {
        if chat(&rig, p, "x").await.backend == Some(0) {
            owner_a.push(p.clone());
        }
    }
    assert!(!owner_a.is_empty(), "with 16 prompts some must belong to replica a");
    a.kill();
    wait_until("a marked down", 3, || !rig.app.backends[0].healthy.load(SeqCst)).await;
    assert_eq!(chat(&rig, &owner_a[0], "x").await.backend, Some(1), "its keys move to the other replica while a is down");
    let a2 = start_mock("a2", Some(addr_a)).await; // same address, as if the process had been restarted
    wait_until("a healthy again", 5, || rig.app.backends[0].healthy.load(SeqCst)).await;
    for p in &owner_a {
        assert_eq!(chat(&rig, p, "x").await.backend, Some(0), "key {p} must go home after the owner recovers");
    }
    assert!(a2.state.hits.load(SeqCst) >= owner_a.len() as u64);
}

#[tokio::test]
async fn server_errors_are_retried_and_the_breaker_opens_then_recovers() {
    let (a, b) = (start_mock("a", None).await, start_mock("b", None).await);
    let rig = start_router(&[&a, &b], PolicyKind::RoundRobin).await;
    a.state.fail.store(true, SeqCst); // answers 503 but still passes health checks: only the breaker can notice
    for i in 0..12 {
        assert_eq!(chat(&rig, "s", &format!("{i}")).await.status, StatusCode::OK, "request {i} must be retried onto replica b");
    }
    let m = metrics(&rig).await;
    assert!(metric(&m, "router_retries_total") >= 3.0);
    assert_eq!(metric(&m, &format!("router_backend_breaker_state{{backend=\"{}\"}}", a.url())), 2.0, "breaker for a must be open:\n{m}");
    let a_hits_while_open = a.state.hits.load(SeqCst);
    for i in 0..6 {
        assert_eq!(chat(&rig, "s", &format!("x{i}")).await.status, StatusCode::OK);
    }
    assert!(a.state.hits.load(SeqCst) <= a_hits_while_open + 1, "an open breaker must stop traffic (at most one probe)");

    a.state.fail.store(false, SeqCst); // the replica recovers
    // Only a real request can probe an open breaker (the health check passes the whole time, so it proves nothing), so keep sending.
    let t0 = Instant::now();
    while rig.app.backends[0].breaker.state() != prefix_router::breaker::State::Closed {
        assert!(t0.elapsed() < Duration::from_secs(5), "breaker never closed after the replica recovered");
        assert_eq!(chat(&rig, "s", "probe").await.status, StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert!(a.state.hits.load(SeqCst) > a_hits_while_open, "the probe must actually have reached replica a");
}

#[tokio::test]
async fn a_replica_that_hangs_is_timed_out_and_retried() {
    let (a, b) = (start_mock("a", None).await, start_mock("b", None).await);
    let rig = start_router(&[&a, &b], PolicyKind::RoundRobin).await;
    a.state.delay_ms.store(5000, SeqCst); // header timeout in the test config is 1500 ms
    let t0 = Instant::now();
    for i in 0..4 {
        assert_eq!(chat(&rig, "s", &format!("{i}")).await.status, StatusCode::OK);
    }
    assert!(t0.elapsed() < Duration::from_secs(12));
    assert!(metric(&metrics(&rig).await, "router_retries_total") >= 1.0);
}

#[tokio::test]
async fn with_every_replica_down_the_router_answers_503_with_a_json_error() {
    let (a, b) = (start_mock("a", None).await, start_mock("b", None).await);
    let rig = start_router(&[&a, &b], PolicyKind::Prefix).await;
    a.kill();
    b.kill();
    wait_until("both marked unhealthy", 3, || rig.app.backends.iter().all(|x| !x.healthy.load(SeqCst))).await;
    let r = chat(&rig, "s", "u").await;
    assert_eq!(r.status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(r.body.contains("router_error"), "{}", r.body);
    let health = rig.client.get(format!("{}/router/health", rig.base)).send().await.unwrap();
    assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn bounded_load_spills_a_hot_prefix_under_concurrency() {
    let (a, b, c) = (start_mock("a", None).await, start_mock("b", None).await, start_mock("c", None).await);
    for m in [&a, &b, &c] {
        m.state.delay_ms.store(150, SeqCst); // keep requests in flight long enough to overlap
    }
    let rig = Arc::new(start_router(&[&a, &b, &c], PolicyKind::Prefix).await);
    let mut tasks = Vec::new();
    for i in 0..30 {
        let rig = rig.clone();
        tasks.push(tokio::spawn(async move { chat(&rig, "ONE shared system prompt", &format!("{i}")).await }));
    }
    let mut per_backend = [0usize; 3];
    for t in tasks {
        let r = t.await.unwrap();
        assert_eq!(r.status, StatusCode::OK);
        per_backend[r.backend.unwrap()] += 1;
    }
    let m = metrics(&rig).await;
    assert!(per_backend.iter().filter(|n| **n > 0).count() >= 2, "a hot prefix must spill beyond one replica: {per_backend:?}");
    assert!(metric(&m, "router_route_decisions_total{decision=\"spill\"}") >= 1.0, "{m}");
    assert!(*per_backend.iter().max().unwrap() < 30, "{per_backend:?}");
}

#[tokio::test]
async fn metrics_report_decisions_and_per_replica_counters() {
    let (a, b) = (start_mock("a", None).await, start_mock("b", None).await);
    let rig = start_router(&[&a, &b], PolicyKind::Prefix).await;
    for i in 0..8 {
        chat(&rig, &format!("p{i}"), "u").await;
    }
    wait_until("all streams accounted for", 3, || rig.app.backends.iter().map(|x| x.stats.ok.load(SeqCst)).sum::<u64>() == 8).await;
    let m = metrics(&rig).await;
    assert_eq!(metric(&m, "router_requests_total"), 8.0);
    assert_eq!(metric(&m, "router_route_decisions_total{decision=\"affinity\"}"), 8.0);
    let ok: f64 = [&a, &b].iter().map(|x| metric(&m, &format!("router_backend_ok_total{{backend=\"{}\"}}", x.url()))).sum();
    assert_eq!(ok, 8.0);
    assert_eq!(metric(&m, "router_request_duration_seconds_count"), 8.0);
    assert_eq!(metric(&m, "router_ttfb_seconds_count"), 8.0);
}

#[tokio::test]
async fn non_chat_requests_are_proxied_too() {
    let a = start_mock("a", None).await;
    let rig = start_router(&[&a], PolicyKind::Prefix).await;
    let r = rig.client.get(format!("{}/health", rig.base)).send().await.unwrap();
    assert_eq!(r.status(), StatusCode::OK, "GET /health falls through to a replica");
}
