"""Read-only: all router metric series names in a snapshot, with the 0.1 s/last values (run from the clone root)."""
import glob
import json

d = json.load(open(sorted(glob.glob("serving/results/raw/router_bench_*.json"))[-1]))
s = d["router_metric_snapshots"][0]["after"]
for k, v in s.items():
    print(k, v)
