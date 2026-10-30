import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_trtllm_view as V

NESTED = {"architectures": ["Qwen2VLForConditionalGeneration"], "dtype": "bfloat16", "model_type": "qwen2_vl", "image_token_id": 151655,
          "text_config": {"hidden_size": 1536, "intermediate_size": 8960, "num_hidden_layers": 28, "num_attention_heads": 12,
                          "num_key_value_heads": 2, "vocab_size": 151936, "rms_norm_eps": 1e-6, "model_type": "qwen2_vl_text", "dtype": "bfloat16",
                          "rope_parameters": {"mrope_section": [16, 24, 24], "rope_theta": 1000000.0, "rope_type": "default", "type": "mrope"}},
          "vision_config": {"depth": 32, "embed_dim": 1280}}


def test_flatten_moves_text_fields_to_the_top_and_keeps_architectures():
    f = V.flatten(NESTED)
    assert f["architectures"] == ["Qwen2VLForConditionalGeneration"] and "text_config" not in f and "dtype" not in f
    assert f["hidden_size"] == 1536 and f["num_key_value_heads"] == 2 and f["torch_dtype"] == "bfloat16"
    assert f["rope_scaling"] == {"type": "mrope", "mrope_section": [16, 24, 24], "rope_type": "default"} and f["rope_theta"] == 1000000.0
    assert f["vision_config"] == NESTED["vision_config"] and f["image_token_id"] == 151655


def test_check_same_shape_passes_for_a_faithful_flatten_and_fails_for_a_changed_field():
    f = V.flatten(NESTED)
    V.check_same_shape(NESTED, f)
    f["num_hidden_layers"] = 27
    with pytest.raises(SystemExit):
        V.check_same_shape(NESTED, f)


def test_the_source_config_is_not_modified():
    import copy
    before = copy.deepcopy(NESTED)
    V.flatten(NESTED)
    assert NESTED == before


def test_tokenizer_config_list_becomes_a_mapping_and_a_mapping_is_kept():
    assert V.tokenizer_config({"extra_special_tokens": ["<|im_start|>"], "eos_token": "<|im_end|>"}) == {"extra_special_tokens": {}, "eos_token": "<|im_end|>"}
    assert V.tokenizer_config({"extra_special_tokens": {"a": "b"}})["extra_special_tokens"] == {"a": "b"}


def test_preprocessor_config_sets_the_pixel_budget_and_keeps_the_normalisation():
    pc = {"image_processor": {"image_mean": [0.1], "image_std": [0.2], "patch_size": 14, "merge_size": 2,
                              "size": {"longest_edge": 12845056, "shortest_edge": 3136}}}
    out = V.preprocessor_config(pc, 1003520)
    assert out["max_pixels"] == 1003520 and out["size"] == {"shortest_edge": 3136, "longest_edge": 1003520} and out["min_pixels"] == 3136
    assert out["image_mean"] == [0.1] and out["patch_size"] == 14 and out["merge_size"] == 2
    assert pc["image_processor"]["size"]["longest_edge"] == 12845056      # the source dict is not modified


def _need_symlinks(tmp_path: Path):
    try:
        (tmp_path / "probe").symlink_to(tmp_path)
    except OSError:
        pytest.skip("this OS account cannot create symlinks (the view script runs in WSL, where these tests run)")
    (tmp_path / "probe").unlink()


def _fake_checkpoint(d: Path):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(NESTED))
    (d / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": ["<|im_start|>"], "eos_token": "<|im_end|>"}))
    (d / "processor_config.json").write_text(json.dumps({"image_processor": {"patch_size": 14, "size": {"longest_edge": 12845056, "shortest_edge": 3136}}}))
    (d / "tokenizer.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"weights")


def _digest(d: Path):
    import hashlib
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.iterdir())}


def test_build_view_never_modifies_the_source_even_when_the_view_holds_stale_symlinks(tmp_path):
    _need_symlinks(tmp_path)
    src, dst = tmp_path / "src", tmp_path / "view"
    _fake_checkpoint(src)
    before = _digest(src)
    # the state an earlier version of the script left behind: symlinks to the source for the files that are later rewritten
    dst.mkdir()
    for name in ("tokenizer_config.json", "processor_config.json", "config.json"):
        (dst / name).symlink_to(src / name)
    assert V.build_view(src, dst, 1003520) is True
    assert _digest(src) == before                                              # the source checkpoint is byte-for-byte unchanged
    assert not (dst / "tokenizer_config.json").is_symlink() and not (dst / "processor_config.json").is_symlink()
    assert json.loads((dst / "tokenizer_config.json").read_text())["extra_special_tokens"] == {}
    assert json.loads((dst / "preprocessor_config.json").read_text())["max_pixels"] == 1003520
    assert (dst / "model.safetensors").is_symlink()                            # weights are a link, not a copy


def test_build_view_can_run_twice(tmp_path):
    _need_symlinks(tmp_path)
    src, dst = tmp_path / "src", tmp_path / "view"
    _fake_checkpoint(src)
    before = _digest(src)
    assert V.build_view(src, dst, 1003520) and V.build_view(src, dst, 1003520)
    assert _digest(src) == before
