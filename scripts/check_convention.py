"""Draw the model's predicted click on synthetic screenshots: ground-truth box (green), click (red cross).
    python scripts/check_convention.py --name clean --adapter runs/clean/adapter
A resize / coordinate-convention bug would show up here as clicks that are systematically shifted or scaled."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from uiclick.evaluate import make_condition, predict, test_samples
from uiclick.model import load_model, load_processor
from uiclick.reward import in_box, parse_click
from uiclick.synth import H, W

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

ss = test_samples("unbiased_test", a.n)
texts = predict(load_model(adapter=a.adapter), load_processor(), *make_condition(ss, "real"), batch_size=8)
cols = 2
sheet = Image.new("RGB", (W * cols, H * ((a.n + 1) // cols)))
agree = 0
for i, (s, t) in enumerate(zip(ss, texts)):
    im = s.image.copy()
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = s.elements[s.target].box
    d.rectangle((x0, y0, x1, y1), outline=(0, 200, 0), width=2)  # pixel-space ground truth, independent of norm_box
    p = parse_click(t)
    if p:
        px, py = p[0] / 1000 * W, p[1] / 1000 * H
        d.line((px - 7, py, px + 7, py), fill=(255, 0, 0), width=3); d.line((px, py - 7, px, py + 7), fill=(255, 0, 0), width=3)
        agree += (x0 <= px <= x1 and y0 <= py <= y1) == in_box(p, s.box)  # pixel-space hit must equal normalised-space hit
    d.text((4, 2), f"{s.instruction} -> {t}", fill=(255, 0, 255))
    sheet.paste(im, ((i % cols) * W, (i // cols) * H))
out = ROOT / "results" / f"convention_synth_{a.name}.png"
sheet.save(out)
print(f"pixel-space vs normalised-space hit agreement: {agree}/{a.n}; wrote {out}")
