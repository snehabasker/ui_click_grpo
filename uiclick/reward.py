"""Verifiable reward: 1 if the click lands inside the target box, else 0.

Anti-hacking rules baked into the parser (each one closes a hole a policy could otherwise exploit):
  * exactly ONE (x, y) pair and NO other number may appear -> no spraying several points to cover the screen,
    and a truncated box like "(733, 414, 886," cannot sneak through as the point (733, 414)
  * coordinates must be within [0, 1000]  -> no off-screen / degenerate answers
  * the box is judged in the model's own 0-1000 space, never in a space the model can inflate
"""
from __future__ import annotations

import re

_SPECIAL = re.compile(r"<\|[^|>]*\|>")
_NUM = r"-?\d+(?:\.\d+)?"
_PAIR = re.compile(rf"\(?\s*({_NUM})\s*,\s*({_NUM})\s*\)?")


def parse_click(text: str):
    """Return (x, y) in 0-1000 space, or None if the answer is not exactly one valid point."""
    text = _SPECIAL.sub(" ", text)
    pairs = _PAIR.findall(text)
    if len(pairs) != 1 or len(re.findall(_NUM, text)) != 2:
        return None
    x, y = float(pairs[0][0]), float(pairs[0][1])
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


def in_box(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def reward(text: str, box, format_bonus: float = 0.0) -> float:
    """1 if the click is inside the box. Unparseable output is exactly 0. ``format_bonus`` (default 0) is added to any
    well-formed answer, hit or miss, so a valid miss scores slightly above garbage."""
    pt = parse_click(text)
    if pt is None:
        return 0.0
    return float(in_box(pt, box)) + format_bonus
