"""Read-only: per-replica successful requests per interleaved router run (the router is restarted per run, so counters are per run)."""
import glob
import json
import sys

f = sorted(glob.glob("serving/results/raw/router_bench_*.json"))[-1] if len(sys.argv) < 2 else sys.argv[1]
d = json.load(open(f))
print(f)
tot = {}
for s in d["router_metric_snapshots"]:
    a = s["after"]
    ok = [a[k] - s["before"].get(k, 0) for k in a if k.startswith("router_backend_ok_total")]
    key = (s["policy"], s["concurrency"])
    t = tot.setdefault(key, [0, 0])
    t[0] += ok[0]
    t[1] += ok[1]
for k, v in sorted(tot.items()):
    print(k, "replica0 =", int(v[0]), "replica1 =", int(v[1]), "share of replica0 = %.3f" % (v[0] / (v[0] + v[1])))
