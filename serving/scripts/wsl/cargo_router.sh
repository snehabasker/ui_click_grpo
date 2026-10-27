#!/bin/bash
# cargo_router.sh <cargo args...>: run cargo for serving/router with the user-space toolchain. Build output goes to the Linux disk
# (not /mnt/c) because compiling over the Windows mount is slow.
export PATH="$HOME/.cargo/bin:$HOME/toolchain/bin:$PATH"
export CC=gcc
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/work/cargo-target}"
SRC="${ROUTER_SRC:-/mnt/c/Users/Sneha Basker/ui-click-rl/serving/router}"
cd "$SRC" || exit 1
exec cargo "$@"
