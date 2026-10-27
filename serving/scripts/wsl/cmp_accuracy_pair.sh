#!/bin/bash
# cmp_accuracy_pair.sh <tagA> <tagB>: paired accuracy comparison on both sets, from the Linux clone's raw results (read-only)
cd "$HOME/work/ui_click_grpo/serving/results/raw" || exit 1
PY="$HOME/venvs/bench/bin/python"
CMP="$HOME/work/ui_click_grpo/serving/scripts/compare_accuracy.py"
[ -f "$CMP" ] || CMP="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/compare_accuracy.py"
for set in screenspot400 synth300; do
  A=$(ls accuracy_${1}_${set}_*.json 2>/dev/null | head -1); B=$(ls accuracy_${2}_${set}_*.json 2>/dev/null | head -1)
  if [ -n "$A" ] && [ -n "$B" ]; then "$PY" "$CMP" "$A" "$B"; else echo "$set: missing a file ($A / $B)"; fi
done
echo "--- ScreenSpot by data type:"
"$PY" - "$1" "$2" <<'PYEOF'
import glob, json, sys
def load(tag):
    return json.load(open(sorted(glob.glob(f"accuracy_{tag}_screenspot400_*.json"))[0]))
a, b = load(sys.argv[1]), load(sys.argv[2])
for k in ("text", "icon"):
    x, y = a["score"]["by_data_type"][k], b["score"]["by_data_type"][k]
    print(f"  {k:5s} {sys.argv[1]} {100*x['acc']:.1f}% -> {sys.argv[2]} {100*y['acc']:.1f}%  (n={x['n']})")
PYEOF
