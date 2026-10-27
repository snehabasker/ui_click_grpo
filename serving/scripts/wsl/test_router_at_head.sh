#!/bin/bash
# Run one router integration test against the COMMITTED router source (git HEAD), not the working tree. usage: test_router_at_head.sh <test filter>
WIN="${WIN_REPO:-/mnt/c/Users/Sneha Basker/ui-click-rl}"
D=$(mktemp -d)
git -C "$WIN" archive HEAD serving/router | tar -x -C "$D"
export ROUTER_SRC="$D/serving/router" CARGO_TARGET_DIR="$HOME/work/cargo-target-head"
bash "$WIN/serving/scripts/wsl/cargo_router.sh" test --release --test integration "$1" 2>&1 | grep -E "^test |test result|panicked"
rm -rf "$D"
