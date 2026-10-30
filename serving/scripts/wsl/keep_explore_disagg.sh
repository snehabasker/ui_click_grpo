#!/usr/bin/env bash
# keep the chunked-prefill diagnostic as labelled exploratory evidence (the bench file was written under the result name before the tag prefix existed)
R="$HOME/work/ui_click_grpo/serving/results/raw"; W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/results/raw"
cd "$R" || exit 1
mv disagg_bench_20261030-181519.json explore_disagg_bench_20261030-181519.json
cp explore_disagg_bench_20261030-181519.json loadtest_explore-disagg-split_c4_r1_20261030-181929.json "$W/"
ls "$W" | grep explore-disagg; ls "$W" | grep explore_disagg
