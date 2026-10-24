"""Full diagnostic suite.  python scripts/eval.py --name base
                         python scripts/eval.py --name clean --adapter runs/clean/adapter
                         python scripts/eval.py --name baselines     (no model)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--cue_eval", action="store_true", help="also run the 2x2 cue-isolation splits (position / ring)")
ap.add_argument("--style_only", action="store_true", help="skip the standard suite (use with --style_eval)")
ap.add_argument("--style_eval", action="store_true", help="also evaluate on seen vs held-out visual styles")
a = ap.parse_args()

out = Path(__file__).resolve().parents[1] / "results" / f"eval_{a.name}.json"
if a.name == "baselines":
    from uiclick.evaluate import run_baselines
    res = run_baselines(a.n)
else:
    from uiclick.evaluate import run_suite
    from uiclick.model import load_model, load_processor
    model = load_model(adapter=a.adapter)
    proc = load_processor()
    res = {} if a.style_only else run_suite(model, proc, a.n, a.bs)
    style = {}
    if a.style_eval:
        for st in ["seen", "heldout"]:
            style[st] = run_suite(model, proc, a.n, a.bs, splits=["unbiased_test", "adversarial_test"], probes=False, style=st)
    if a.cue_eval:
        from uiclick.evaluate import CUE_SPLITS
        res.update(run_suite(model, proc, a.n, a.bs, splits=CUE_SPLITS, probes=False))
from uiclick.meta import run_meta
out.write_text(json.dumps({"name": a.name, "adapter": a.adapter, "n": a.n, "results": res, "style_eval": locals().get("style", {}),
                           "meta": run_meta({"eval_args": vars(a)}) if a.name != "baselines" else {}}, indent=1))
print("wrote", out)
