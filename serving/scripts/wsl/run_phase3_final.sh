#!/usr/bin/env bash
set -u
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"
for s in sync_clone run_kernel_tests run_kernel_bench; do sed 's/\r$//' "$W/$s.sh" > /tmp/$s.sh; done
bash /tmp/sync_clone.sh || exit 1
bash /tmp/run_kernel_tests.sh | tail -2 || exit 1
mkdir -p ~/work/logs
bash /tmp/run_kernel_bench.sh --reps 5 --tag rmsnorm-int8 2>&1 | tee ~/work/logs/kernel_bench.log | tail -12
