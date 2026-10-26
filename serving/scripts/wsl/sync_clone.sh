#!/bin/bash
# Bring the Linux clone to the Windows checkout's latest COMMITTED state, without going through GitHub.
# Raw results record `repo_commit`, so benchmarks must run committed code; uncommitted changes are not synced on purpose.
set -euo pipefail
WIN="${WIN_REPO:-/mnt/c/Users/Sneha Basker/ui-click-rl}"
DST="$HOME/work/ui_click_grpo"
git config --global --add safe.directory "$WIN/.git" 2>/dev/null || true
git config --global --add safe.directory "$WIN" 2>/dev/null || true
git -C "$DST" pull --ff-only "$WIN" main 2>&1 | tail -3
echo "clone HEAD: $(git -C "$DST" rev-parse --short HEAD)   windows HEAD: $(git -C "$WIN" rev-parse --short HEAD)"
git -C "$DST" status --short | grep -v '^??' | head -5 || true
