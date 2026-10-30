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
