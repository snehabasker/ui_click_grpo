"""GRPO training.  e.g.  python scripts/train.py --train_split clean --out runs/clean --steps 150"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uiclick.grpo import train

ap = argparse.ArgumentParser()
ap.add_argument("--train_split", default="clean", choices=["clean", "biased"],
                help="clean: cues independent of target. biased: target is left / ringed 90% of the time.")
ap.add_argument("--train_style", default="all", choices=["all", "seen"],
                help="seen: train only on themes/fonts that exclude the held-out style (an 'unseen app' test)")
ap.add_argument("--format_bonus", type=float, default=0.0, help="extra reward for a parseable answer (0 = off)")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--prompts_per_step", type=int, default=4)
ap.add_argument("--group_size", type=int, default=8)
ap.add_argument("--max_new_tokens", type=int, default=24)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--beta", type=float, default=0.0, help="KL weight to the base model (0 = off)")
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--max_grad_norm", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--eval_n", type=int, default=48)
ap.add_argument("--eval_bs", type=int, default=8)
ap.add_argument("--save_every", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
train(ap.parse_args())
