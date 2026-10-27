#!/bin/bash
# Rust toolchain WITHOUT sudo (rustup installs into ~/.cargo and ~/.rustup). Rust links through a C compiler; this Ubuntu image has
# none, so point cargo at the user-space GCC from 04_install_toolchain.sh and prove the combination works with a build + run.
set -uo pipefail
if [ ! -x "$HOME/.cargo/bin/rustc" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
export PATH="$HOME/.cargo/bin:$HOME/toolchain/bin:$PATH"
mkdir -p "$HOME/.cargo"
cat > "$HOME/.cargo/config.toml" <<'CFG'
[target.x86_64-unknown-linux-gnu]
linker = "gcc"
CFG
rustc --version
cargo --version
# smoke test: a tiny program that uses std, threads and the network stack, built and run
T=$(mktemp -d)
cd "$T" && cargo new --quiet smoke >/dev/null 2>&1 && cd smoke
cat > src/main.rs <<'RS'
use std::net::TcpListener;
fn main() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let h = std::thread::spawn(|| 40 + 2);
    println!("rust ok: thread={} port_bound={}", h.join().unwrap(), l.local_addr().unwrap().port() > 0);
}
RS
cargo run --quiet 2>&1 | tail -3
echo "RUST_READY"
