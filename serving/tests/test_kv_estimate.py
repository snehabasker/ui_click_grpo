import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import kv_transfer_estimate as E

CFG = {"text_config": {"num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2, "hidden_size": 1536}}
STATS = {"shared_prefix_tokens_approx": 1254, "tokens_first_turn_with_prefix": 1666, "tokens_last_turn": 1861}


def test_kv_bytes_per_token_for_the_models_shape():
    assert E.kv_bytes_per_token(28, 2, 128, 2) == 28672          # 2 x 28 x 2 x 128 x 2 bytes
    assert E.kv_bytes_per_token(28, 2, 128, 1) == 14336


def test_theoretical_link_rates():
    assert abs(E.LINKS["PCIe 4.0 x16"][0] - 31.508) < 0.01       # 16 GT/s x 16 lanes x 128/130 / 8
    assert abs(E.LINKS["PCIe 3.0 x16"][0] - 15.754) < 0.01
    assert E.LINKS["10 GbE"][0] == 1.25 and E.LINKS["100 GbE"][0] == 12.5


def test_transfer_time_is_bytes_over_rate():
    assert E.transfer_ms(1e9, 10.0) == 100.0


def test_build_derives_head_dim_and_rows():
    rec = E.build(CFG, STATS, 700.0)
    assert rec["model"]["head_dim"] == 128 and rec["bytes_per_token"] == {"bf16": 28672, "fp8": 14336}
    assert len(rec["rows"]) == 3 * 2 * len(E.LINKS)
    row = next(r for r in rec["rows"] if r["tokens"] == 1666 and r["kv_dtype"] == "bf16" and r["link"] == "10 GbE")
    assert row["kv_bytes"] == 1666 * 28672
    assert abs(row["ms_at_line_rate"] - 1666 * 28672 / 1.25e9 * 1e3) < 1e-9
    assert abs(row["fraction_of_reference_ttft"] - row["ms_at_line_rate"] / 700.0) < 1e-12


def test_markdown_is_labelled_an_estimate():
    md = E.markdown(E.build(CFG, STATS, 700.0))
    assert md.startswith("**ESTIMATE, not a measurement.**") and "28,672 bytes (bf16)" in md
