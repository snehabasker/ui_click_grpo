"""Everything needed to reproduce a run, written next to its outputs."""
from __future__ import annotations

import datetime
import platform
import subprocess

from .synth import FONTS, SPLITS, TEST_SEED_BASE, THEMES, W, H


def run_meta(extra: dict | None = None) -> dict:
    import peft
    import torch
    import transformers

    from .model import MODEL_ID, PROMPT

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        git = None
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_id": MODEL_ID, "prompt": PROMPT, "image_size": [W, H],
        "coordinate_convention": "model answers (x, y) in 0-1000 normalised image coords; boxes are x0,y0,x1,y1 in the same space",
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__, "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "git_commit": git,
        "data": {"splits": {k: list(v) for k, v in SPLITS.items()}, "test_seed_base": TEST_SEED_BASE,
                 "fonts_available": FONTS, "n_themes": len(THEMES)},
        **(extra or {}),
    }
