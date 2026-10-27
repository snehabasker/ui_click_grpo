"""Read-only: the failed requests, retry counters and run length of the failure test (run from the repo root).
The failure test's loadtest file, router_failure json (timeline) and router counters."""
import glob
import json

lt = json.load(open(sorted(glob.glob("serving/results/raw/loadtest_router-failure_*.json"))[-1]))
print("wall_s", lt["wall_s"], "summary:", {k: lt["summary"][k] for k in ("requests", "ok", "errors", "req_per_s")})
t0 = min(r.get("t_start", 0) for r in lt["records"])
print("record keys:", list(lt["records"][0]))
for r in lt["records"]:
    if not r.get("ok", True) or r.get("error"):
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in
               ("t_start", "t_end", "ttft", "e2e", "ok", "error", "status", "completion_tokens", "session", "turn")}, "t_rel", round(r.get("t_start", 0) - t0, 1))
fj = json.load(open(sorted(glob.glob("serving/results/raw/router_failure_*.json"))[-1]))
print("failure keys:", list(fj))
tl = fj.get("timeline")
if tl:
    print("timeline samples:", len(tl), "first:", tl[0], "last:", tl[-1])
for k in ("router_metrics_final", "router_metrics_after"):
    if k in fj:
        print(k, {a: b for a, b in fj[k].items() if "retries" in a or "errors" in a or "aborted" in a or "retried" in a or "ok_total" in a or "went_down" in a or "came_up" in a})
