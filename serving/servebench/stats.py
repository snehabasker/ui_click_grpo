"""Small, dependency-light statistics helpers (no scipy)."""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median_range(xs: list[float]) -> dict:
    """Median and min-max spread over repeats. With few repeats the range is more honest than a std."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(xs), "median": percentile(xs, 50), "min": min(xs), "max": max(xs)}


def sign_test_p(b: int, c: int) -> float:
    """Exact two-sided sign test on discordant pairs (b: only A correct, c: only B correct)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
