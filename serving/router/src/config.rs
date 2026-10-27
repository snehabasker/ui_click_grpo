use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum PolicyKind {
    /// Consistent hashing on the prompt prefix with bounded load, falling back to least-loaded.
    Prefix,
    RoundRobin,
    LeastLoaded,
}

#[derive(Parser, Clone, Debug)]
#[command(name = "prefix-router", version, about = "Prefix-aware router for OpenAI-compatible LLM servers")]
pub struct Config {
    #[arg(long, default_value = "127.0.0.1:9000")]
    pub listen: String,

    /// Replica base URL, e.g. http://127.0.0.1:8001 (repeat the flag for each replica).
    #[arg(long = "backend", required = true)]
    pub backends: Vec<String>,

    #[arg(long, value_enum, default_value_t = PolicyKind::Prefix)]
    pub policy: PolicyKind,

    /// Bounded load: a replica may hold at most ceil(load_factor * average in-flight) requests before its keys spill over.
    #[arg(long, default_value_t = 1.25)]
    pub load_factor: f64,

    /// Virtual nodes per replica on the hash ring.
    #[arg(long, default_value_t = 160)]
    pub vnodes: usize,

    /// Hash the first N messages as the prefix key. Default: only the system message(s), which agents share.
    #[arg(long)]
    pub prefix_messages: Option<usize>,

    /// Retries on another replica when an attempt fails BEFORE the response starts streaming.
    #[arg(long, default_value_t = 2)]
    pub max_retries: u32,
    #[arg(long, default_value_t = 50)]
    pub backoff_base_ms: u64,
    #[arg(long, default_value_t = 1000)]
    pub backoff_max_ms: u64,

    #[arg(long, default_value_t = 1000)]
    pub connect_timeout_ms: u64,
    /// Time allowed for a replica to start answering (response headers).
    #[arg(long, default_value_t = 30000)]
    pub header_timeout_ms: u64,
    /// Longest silence allowed between chunks of a streaming response.
    #[arg(long, default_value_t = 30000)]
    pub idle_timeout_ms: u64,

    #[arg(long, default_value = "/health")]
    pub health_path: String,
    #[arg(long, default_value_t = 500)]
    pub health_interval_ms: u64,
    #[arg(long, default_value_t = 800)]
    pub health_timeout_ms: u64,
    #[arg(long, default_value_t = 2)]
    pub unhealthy_after: u32,
    #[arg(long, default_value_t = 2)]
    pub healthy_after: u32,

    #[arg(long, default_value_t = 5)]
    pub breaker_threshold: u32,
    #[arg(long, default_value_t = 3000)]
    pub breaker_cooldown_ms: u64,

    #[arg(long, default_value_t = 64)]
    pub max_body_mb: usize,
}

impl Config {
    /// Fast timings for tests.
    pub fn for_tests(backends: &[String], policy: PolicyKind) -> Self {
        Self::for_tests_with(backends, policy, &[])
    }

    /// Fast timings with some values overridden, e.g. `&[("--health-interval-ms", "60000")]` to keep the health checker out of the way.
    pub fn for_tests_with(backends: &[String], policy: PolicyKind, overrides: &[(&str, &str)]) -> Self {
        let mut args: Vec<String> = vec!["prefix-router".into()];
        for b in backends {
            args.push("--backend".into());
            args.push(b.clone());
        }
        let policy = match policy {
            PolicyKind::Prefix => "prefix",
            PolicyKind::RoundRobin => "round-robin",
            PolicyKind::LeastLoaded => "least-loaded",
        };
        let mut defaults: Vec<(&str, &str)> = vec![
            ("--policy", policy),
            ("--health-interval-ms", "40"),
            ("--health-timeout-ms", "300"),
            ("--breaker-threshold", "3"),
            ("--breaker-cooldown-ms", "250"),
            ("--backoff-base-ms", "5"),
            ("--backoff-max-ms", "40"),
            ("--connect-timeout-ms", "300"),
            ("--header-timeout-ms", "1500"),
        ];
        for (k, v) in overrides {
            match defaults.iter_mut().find(|d| d.0 == *k) {
                Some(d) => d.1 = v,
                None => defaults.push((k, v)),
            }
        }
        for (k, v) in defaults {
            args.push(k.into());
            args.push(v.into());
        }
        Config::parse_from(args)
    }
}
