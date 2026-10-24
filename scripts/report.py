"""Build figures + results/results.md from results/eval_*.json and runs/*/train_log.jsonl."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES, RUNS = ROOT / "results", ROOT / "runs"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = {  # fixed entity -> colour (validated categorical slots 1-3), never re-assigned
    "base": ("Base Qwen2-VL-2B", "#2a78d6"),
    "clean": ("GRPO, clean data", "#eb6834"),
    "biased": ("GRPO, biased data (90% cues)", "#1baf7a"),
    "holdout": ("GRPO, clean, unseen-style holdout + format bonus", "#e87ba4"),
}
SPLITS = [("unbiased_test", "Unbiased\n(no cue)"), ("iid_biased_test", "IID-biased\n(cues help)"),
          ("adversarial_test", "Adversarial\n(cues mislead)")]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True})


def load():
    ev = {}
    for k in list(MODELS) + ["baselines"]:
        f = RES / f"eval_{k}.json"
        if f.exists():
            ev[k] = json.loads(f.read_text())["results"]
    logs = {}
    for k in ["clean", "biased", "holdout"]:
        f = RUNS / k / "train_log.jsonl"
        if f.exists():
            logs[k] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return ev, logs


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def fig_shortcut(ev):
    models = [m for m in MODELS if m in ev]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        accs, lo, hi = [], [], []
        for s, _ in SPLITS:
            r = ev[m][s]["real"]
            accs.append(100 * r["acc"]); lo.append(100 * (r["acc"] - r["ci"][0])); hi.append(100 * (r["ci"][1] - r["acc"]))
        x = np.arange(len(SPLITS)) + (i - (len(models) - 1) / 2) * w
        ax.bar(x, accs, w * 0.92, color=MODELS[m][1], label=MODELS[m][0], yerr=[lo, hi],
               error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
        for xi, a, h in zip(x, accs, hi):
            ax.text(xi, a + h + 1.5, f"{a:.0f}", ha="center", fontsize=9, color=INK)
    if "baselines" in ev:
        for j, (s, _) in enumerate(SPLITS):
            v = 100 * ev["baselines"][s]["ring_oracle"]["acc"]
            ax.plot([j - 0.42, j + 0.42], [v, v], color=INK, lw=1.2, ls="--")
        ax.plot([], [], color=INK, lw=1.2, ls="--", label="'Click the ringed element' (no model)")
    ax.set_xticks(range(len(SPLITS)), [n for _, n in SPLITS])
    ax.set_ylabel("Click accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("Does RL help, and did it learn the shortcut?  (95% Wilson CIs)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.0), ncol=1)
    fig.tight_layout(); fig.savefig(RES / "fig_shortcut_gap.png", dpi=160); plt.close(fig)


def fig_training(logs):
    if not logs:
        return
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for k, log in logs.items():
        st = np.array([r["step"] for r in log])
        rw = np.array([r["reward"] for r in log])
        sm = np.convolve(np.pad(rw, (4, 0), mode="edge"), np.ones(5) / 5, mode="valid")
        axs[0].plot(st, 100 * sm, color=MODELS[k][1], lw=2, label=MODELS[k][0])
    axs[0].set_title("Train reward (sampled, 5-step avg)", loc="left", fontsize=10); axs[0].set_xlabel("step"); axs[0].set_ylabel("%")
    axs[0].legend(frameon=False, fontsize=8.5)
    for ax, k in zip(axs[1:], ["clean", "biased", "holdout"]):
        if k not in logs:
            continue
        ev = [r for r in logs[k] if "eval_unbiased_test_acc" in r]
        st = [r["step"] for r in ev]
        ax.plot(st, [100 * r["eval_unbiased_test_acc"] for r in ev], color=MODELS[k][1], lw=2, marker="o", ms=4, label="unbiased test" + (" (seen style)" if k == "holdout" else ""))
        second, lab = (("eval_unbiased_heldout_style_acc", "unbiased test, UNSEEN style") if k == "holdout" else ("eval_adversarial_test_acc", "adversarial test"))
        ax.plot(st, [100 * r[second] for r in ev], color=MODELS[k][1], lw=2, ls="--", marker="s", ms=4, label=lab)
        ax.set_title(f"Held-out accuracy: {k} run (n=48/point, noisy)", loc="left", fontsize=10); ax.set_xlabel("step"); ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(RES / "fig_training.png", dpi=160); plt.close(fig)


def fig_probes(ev):
    models = [m for m in MODELS if m in ev]
    probes = [("real", "Real\ninstruction"), ("shuffled_instr", "Wrong\ninstruction"), ("blank_image", "Blank\nimage")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (s, name) in zip(axs, [SPLITS[0], SPLITS[2]]):
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            vals = [100 * ev[m][s][p]["acc"] for p, _ in probes]
            x = np.arange(len(probes)) + (i - (len(models) - 1) / 2) * w
            ax.bar(x, vals, w * 0.92, color=MODELS[m][1], label=MODELS[m][0])
            for xi, v in zip(x, vals):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(probes)), [n for _, n in probes])
        ax.set_title(name.replace("\n", " "), loc="left", fontsize=10)
    axs[0].set_ylabel("Click accuracy (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Instruction / image ablations: accuracy that survives them is not grounded in the instruction", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(RES / "fig_probes.png", dpi=160); plt.close(fig)


def _load(name):
    f = RES / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def ci(r):
    return f"{pct(r['acc'])} [{pct(r['ci'][0])}, {pct(r['ci'][1])}]"


def extra_tables() -> list:
    L = []
    # ---- cue isolation
    rows = []
    for m in ["clean", "biased"]:
        d = _load(f"eval_{m}_cue")
        if not d:
            continue
        r = d["results"]
        pa, pn, ra, rn = (r[k]["real"] for k in ["pos_aligned_test", "pos_anti_test", "ring_aligned_test", "ring_anti_test"])
        rows.append(f"| {MODELS[m][0]} | {ci(pa)} | {ci(pn)} | {100 * (pa['acc'] - pn['acc']):+.1f} pts | {ci(ra)} | {ci(rn)} | {100 * (ra['acc'] - rn['acc']):+.1f} pts |")
    if rows:
        L += ["\n## Cue isolation (one cue aligned or anti-aligned with the target, the other independent)\n",
              "| model | position: target in left half | position: target in right half | position effect | ring: target ringed | ring: target not ringed | ring effect |",
              "|---|---|---|---|---|---|---|"] + rows
    # ---- unseen style
    rows = []
    for m, nm in [("base_style", "Base Qwen2-VL-2B"), ("holdout", MODELS["holdout"][0])]:
        d = _load(f"eval_{m}")
        if d and d.get("style_eval"):
            se = d["style_eval"]
            rows.append(f"| {nm} | " + " | ".join(ci(se[st][sp]["real"]) for st in ["seen", "heldout"] for sp in ["unbiased_test", "adversarial_test"]) + " |")
    if rows:
        L += ["\n## Unseen visual style (a colour theme + font never used in training)\n",
              "| model | seen style, unbiased | seen style, adversarial | UNSEEN style, unbiased | UNSEEN style, adversarial |", "|---|---|---|---|---|"] + rows
    # ---- ScreenSpot
    rows, base = [], _load("screenspot_baselines")
    for m in ["base", "clean", "biased", "holdout"]:
        d = _load(f"screenspot_{m}")
        if d and "model" in d:
            r = d["model"]
            rows.append(f"| {MODELS[m][0]} | {ci(r)} | {pct(r['valid_format'])} | " + " | ".join(pct(r["by_data_type"][k]["acc"]) for k in ["text", "icon"]) + " | " +
                        " | ".join(pct(r["by_size"][z]["acc"]) for z in ["small", "medium", "large"]) + " |")
            ref = d
    if rows:
        c = ref["baseline_center"]
        rows.append(f"| *baseline: always click centre* | {ci(c)} | | " + " | ".join(pct(c["by_data_type"][k]["acc"]) for k in ["text", "icon"]) + " | " + " | ".join(pct(c["by_size"][z]["acc"]) for z in ["small", "medium", "large"]) + " |")
        L += [f"\n## Real UIs: ScreenSpot zero-shot transfer (n = {ref['model']['n']}, never trained on)\n",
              "| model | accuracy [95% CI] | valid format | text | icon | small | medium | large |", "|---|---|---|---|---|---|---|---|"] + rows
    return L


def tables(ev, logs) -> str:
    L = ["# Results (auto-generated by scripts/report.py)\n"]
    models = [m for m in MODELS if m in ev]
    n = next(iter(ev[models[0]].values()))["real"]["n"] if models else "?"
    L += [f"## Accuracy (n = {n} per cell, 95% Wilson CI)\n", "| model | " + " | ".join(s.replace("_test", "") for s, _ in SPLITS) + " | shortcut gap (iid − adv) |",
          "|---|" + "---|" * (len(SPLITS) + 1)]
    for m in models:
        cells = [f"{pct(ev[m][s]['real']['acc'])} [{pct(ev[m][s]['real']['ci'][0])}, {pct(ev[m][s]['real']['ci'][1])}]" for s, _ in SPLITS]
        gap = ev[m]["iid_biased_test"]["real"]["acc"] - ev[m]["adversarial_test"]["real"]["acc"]
        L.append(f"| {MODELS[m][0]} | " + " | ".join(cells) + f" | {100 * gap:+.1f} pts |")
    if "baselines" in ev:
        for b, label in [("random_point", "random click"), ("center", "always click centre"), ("train_prior", "always click mean train target"),
                         ("random_element", "click a random element"), ("ring_oracle", "always click the ringed element")]:
            cells = [pct(ev["baselines"][s][b]["acc"]) for s, _ in SPLITS]
            L.append(f"| *baseline: {label}* | " + " | ".join(cells) + " | |")
    L += ["\n## Ablation probes (accuracy against the original target box)\n", "| model | split | real | wrong instruction | blank image |", "|---|---|---|---|---|"]
    for m in models:
        for s in ["unbiased_test", "adversarial_test"]:
            r = ev[m][s]
            L.append(f"| {MODELS[m][0]} | {s.replace('_test', '')} | {pct(r['real']['acc'])} | {pct(r['shuffled_instr']['acc'])} | {pct(r['blank_image']['acc'])} |")
    L += ["\n## Where do clicks go? (adversarial split: ringed decoy never is the target)\n",
          "| model | clicks on ringed decoy | clicks in left half | targets in left half | valid-format | most common click (share) |", "|---|---|---|---|---|---|"]
    for m in models:
        r = ev[m]["adversarial_test"]["real"]
        L.append(f"| {MODELS[m][0]} | {pct(r['decoy_rate'])} | {pct(r.get('left_click_rate'))} | {pct(r['target_left_rate'])} | {pct(r['valid_format'])} | {pct(r.get('top_click_share'))} |")
    L += ["\n## Accuracy by target type and size (unbiased split)\n"]
    kinds = sorted(ev[models[0]]["unbiased_test"]["real"]["by_kind"]) if models else []
    L += ["| model | " + " | ".join(kinds) + " | small | medium | large |", "|---|" + "---|" * (len(kinds) + 3)]
    for m in models:
        r = ev[m]["unbiased_test"]["real"]
        L.append(f"| {MODELS[m][0]} | " + " | ".join(pct(r["by_kind"][k]["acc"]) for k in kinds) + " | " +
                 " | ".join(pct(r["by_size"][z]["acc"]) for z in ["small", "medium", "large"]) + " |")
    L += extra_tables()
    if logs:
        L += ["\n## Training summary\n", "| run | steps | first-10 reward | last-10 reward | informative groups (last 10) | valid format (last 10) | peak VRAM | wall time |", "|---|---|---|---|---|---|---|---|"]
        for k, log in logs.items():
            f10, l10 = log[:10], log[-10:]
            avg = lambda rs, key: np.mean([r[key] for r in rs if r.get(key) is not None])
            L.append(f"| {k} | {len(log)} | {pct(avg(f10, 'reward'))} | {pct(avg(l10, 'reward'))} | {pct(avg(l10, 'informative_frac'))} | "
                     f"{pct(avg(l10, 'valid_format'))} | {log[-1].get('peak_gb', 0):.1f} GB | {log[-1]['elapsed'] / 60:.0f} min |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ev, logs = load()
    if any(m in ev for m in MODELS):
        fig_shortcut(ev); fig_probes(ev)
    fig_training(logs)
    (RES / "results.md").write_text(tables(ev, logs), encoding="utf-8")
    print("wrote results/results.md and figures")
