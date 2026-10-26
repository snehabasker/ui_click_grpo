import json

import pytest
from PIL import Image

from servebench import envinfo
from servebench.data import Item, load_manifest, save_png, verify_manifest, write_manifest
from servebench.stats import median_range, percentile, sign_test_p, wilson


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4, 5], 0) == 1 and percentile([1, 2, 3, 4, 5], 100) == 5


def test_median_range_ignores_none():
    assert median_range([3, 1, 2, None]) == {"n": 3, "median": 2, "min": 1, "max": 3}
    assert median_range([])["median"] is None


def test_wilson_matches_known_values():
    lo, hi = wilson(190, 300)          # 63.3% -> the interval reported in the main README
    assert 0.577 < lo < 0.579 and 0.685 < hi < 0.687
    assert wilson(0, 0) == (0.0, 0.0)


def test_sign_test():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0
    assert sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert sign_test_p(3, 4) > 0.9


def test_manifest_roundtrip_and_tamper_detection(tmp_path):
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    h = save_png(img, tmp_path / "images" / "a.png")
    write_manifest(tmp_path, [Item("a", "images/a.png", h, "click it", [0, 0, 10, 10], {"kind": "button"})], {"name": "t"})
    meta, items = load_manifest(tmp_path)
    assert meta["name"] == "t" and items[0].box == [0, 0, 10, 10]
    assert verify_manifest(tmp_path) == []
    (tmp_path / "images" / "a.png").write_bytes(b"corrupted")
    assert verify_manifest(tmp_path) == ["a"]


def test_model_hash_is_stable_and_content_sensitive(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "w.safetensors").write_bytes(b"abc")
    h1 = envinfo.model_hash(tmp_path)
    assert h1 == envinfo.model_hash(tmp_path)                   # cached
    (tmp_path / "MODEL_HASH").unlink()
    (tmp_path / "w.safetensors").write_bytes(b"abd")
    assert envinfo.model_hash(tmp_path) != h1


def test_capture_has_required_fields(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    env = envinfo.capture(tmp_path, {"seed": 3, "concurrency": 4})
    for k in ("date", "command", "gpu", "packages", "model_hash", "repo_commit", "seed", "concurrency", "python", "platform"):
        assert k in env
    assert env["seed"] == 3
    json.dumps(env)                                              # must be serialisable


def test_scrub_removes_the_home_directory_in_every_spelling():
    from pathlib import Path
    home = str(Path.home())
    obj = {"native": home + "/work/model", "fwd": home.replace("\\", "/") + "/work", "nested": [{"p": home}], "keep": "relative/path"}
    out = envinfo.dumps_scrubbed(obj)
    assert home not in out and home.replace("\\", "\\\\") not in out and home.replace("\\", "/") not in out
    assert "~/work/model" in out and "relative/path" in out
    json.loads(out)                                       # still valid JSON
