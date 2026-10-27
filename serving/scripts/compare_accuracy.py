"""Paired comparison of two accuracy runs on the same items (rule 3): did a serving change move click accuracy?

    python serving/scripts/compare_accuracy.py <accuracy_A.json> <accuracy_B.json>
Prints both accuracies, the discordant counts (items only A / only B got right) and an exact two-sided sign test. A paired test on the
same items is more sensitive than comparing two confidence intervals."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving"))

from servebench.stats import sign_test_p

a, b = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
assert a["set"] == b["set"], "runs are on different sets"
ha, hb = a["score"]["per_item_hit"], b["score"]["per_item_hit"]
assert len(ha) == len(hb)
only_a = sum(x and not y for x, y in zip(ha, hb))
only_b = sum(y and not x for x, y in zip(ha, hb))
p = sign_test_p(only_a, only_b)
print(f"{a['set']}: A={a['tag']} {100*a['score']['acc']:.2f}%  B={b['tag']} {100*b['score']['acc']:.2f}%  "
      f"delta(B-A)={100*(b['score']['acc']-a['score']['acc']):+.2f} pts | only A right {only_a}, only B right {only_b}, sign-test p={p:.4f}")
