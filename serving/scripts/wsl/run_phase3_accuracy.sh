#!/usr/bin/env bash
# sync the clone to the committed HEAD, then run the kernel accuracy check (log in ~/work/logs)
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"
for s in sync_clone run_kernel_accuracy; do sed 's/\r$//' "$W/$s.sh" > /tmp/$s.sh; done
bash /tmp/sync_clone.sh || exit 1
mkdir -p ~/work/logs
bash /tmp/run_kernel_accuracy.sh 2>&1 | tee ~/work/logs/kernel_accuracy.log | tail -25
