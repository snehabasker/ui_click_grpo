use clap::Parser;
use prefix_router::{build_app, health, Config};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();
    let cfg = Config::parse();
    let listen = cfg.listen.clone();
    tracing::info!(policy = ?cfg.policy, backends = ?cfg.backends, load_factor = cfg.load_factor, "starting router on {listen}");
    let (router, app) = build_app(cfg.clone());
    let _health = health::spawn(&app.backends, &cfg);
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
}
