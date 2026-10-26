#!/bin/bash
# Copy the git-ignored artifacts from the Windows checkout into the Linux clone and verify every checksum there.
# Images must come from Windows (fonts!), and reading 4 GB of weights through /mnt/c on every engine start would add I/O noise.
set -euo pipefail
WIN="${WIN_REPO:-/mnt/c/Users/Sneha Basker/ui-click-rl}"
DST="$HOME/work/ui_click_grpo"
PY="$HOME/venvs/bench/bin/python"

mkdir -p "$DST/serving/models" "$DST/serving/data/synth300" "$DST/serving/data/screenspot400" "$DST/serving/traces/default"
[ -d "$DST/serving/models/merged-clean" ] || cp -r "$WIN/serving/models/merged-clean" "$DST/serving/models/merged-clean"
for s in synth300 screenspot400; do
  [ -d "$DST/serving/data/$s/images" ] || cp -r "$WIN/serving/data/$s/images" "$DST/serving/data/$s/images"
done
[ -d "$DST/serving/traces/default/images" ] || cp -r "$WIN/serving/traces/default/images" "$DST/serving/traces/default/images"

cd "$DST"
"$PY" - <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "serving")
from servebench.data import verify_manifest
from servebench import envinfo

root = Path("serving")
ok = True
for s in ("synth300", "screenspot400"):
    bad = verify_manifest(root / "data" / s)
    print(f"manifest {s}: {'OK' if not bad else f'{len(bad)} BAD: {bad[:3]}'}")
    ok &= not bad

tdir = root / "traces" / "default"
bad = 0
n = 0
for line in (tdir / "trace.jsonl").read_text().splitlines():
    s = json.loads(line)
    for it in s["fewshot"] + s["turns"]:
        n += 1
        if hashlib.sha256((tdir / it["image"]).read_bytes()).hexdigest() != it["sha256"]:
            bad += 1
print(f"trace images: {n - bad}/{n} match their sha256")
ok &= bad == 0

m = root / "models" / "merged-clean"
cached = (m / "MODEL_HASH").read_text().strip()      # hash computed on Windows
(m / "MODEL_HASH").unlink()                          # force recomputation over the bytes that were actually copied
fresh = envinfo.model_hash(m)
print(f"model hash on Linux copy: {fresh}\nmodel hash recorded on Windows: {cached}\nmatch: {fresh == cached}")
ok &= fresh == cached
print("ARTIFACTS_OK" if ok else "ARTIFACTS_MISMATCH")
sys.exit(0 if ok else 1)
PYEOF
du -sh "$DST/serving/models" "$DST/serving/data" "$DST/serving/traces" | sed 's/\t/  /'
