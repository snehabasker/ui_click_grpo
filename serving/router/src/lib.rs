//! Prefix-aware router for OpenAI-compatible LLM servers. See README.md for the design and its limits.
pub mod backend;
pub mod breaker;
pub mod config;
pub mod health;
pub mod metrics;
pub mod policy;
pub mod prefix;
pub mod proxy;
pub mod ring;

pub use config::{Config, PolicyKind};
pub use proxy::{build_app, App};
