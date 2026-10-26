#!/bin/bash
# Bring the Linux clone to the Windows checkout's latest COMMITTED state, without going through GitHub.
# Raw results record `repo_commit`, so benchmarks must run committed code; uncommitted changes are not synced on purpose.
#
# Fails loudly (exit 1, "SYNC_FAILED") unless the clone ends up at exactly the Windows HEAD. An earlier version piped the pull
# through `tail`, which hid a failed pull: results files that exist untracked in the clone and also in the incoming commits make
# git refuse the merge, and the clone silently stayed on old code.
set -uo pipefail
WIN="${WIN_REPO:-/mnt/c/Users/Sneha Basker/ui-click-rl}"
DST="$HOME/work/ui_click_grpo"
git config --global --add safe.directory "$WIN/.git" 2>/dev/null || true
git config --global --add safe.directory "$WIN" 2>/dev/null || true

# Untracked raw results in the clone that are byte-identical to the copy already in the Windows checkout are duplicates (they were
# collected and committed there); remove them so the pull can bring in the tracked copies. Anything else is kept.
removed=0
while IFS= read -r f; do
  b=$(basename "$f")
  if [ -e "$WIN/serving/results/raw/$b" ] && cmp -s "$DST/$f" "$WIN/serving/results/raw/$b"; then
    rm -f "$DST/$f"; removed=$((removed + 1))
  fi
done < <(git -C "$DST" ls-files --others --exclude-standard -- serving/results/raw)
echo "removed $removed duplicate raw result files from the clone"
# Generated tables (summarize.py output) are derived and rebuilt on demand; an untracked copy in the clone must not block the pull.
git -C "$DST" ls-files --others --exclude-standard -- serving/results/tables | while IFS= read -r f; do rm -f "$DST/$f"; done

if ! git -C "$DST" pull --ff-only "$WIN" main; then
  echo "SYNC_FAILED: git pull did not complete"; exit 1
fi
CLONE_HEAD=$(git -C "$DST" rev-parse HEAD)
WIN_HEAD=$(git -C "$WIN" rev-parse HEAD)
if [ "$CLONE_HEAD" != "$WIN_HEAD" ]; then
  echo "SYNC_FAILED: clone is at ${CLONE_HEAD:0:7}, Windows is at ${WIN_HEAD:0:7}"; exit 1
fi
dirty=$(git -C "$DST" status --short | grep -v '^??' | wc -l)
[ "$dirty" != "0" ] && echo "warning: $dirty tracked files differ from HEAD in the clone"
echo "SYNC_OK ${CLONE_HEAD:0:7}"
