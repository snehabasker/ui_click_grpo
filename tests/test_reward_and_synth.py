import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uiclick.reward import parse_click, reward
from uiclick.synth import SPLITS, TEST_SEED_BASE, W, H, make_sample

BOX = (100, 100, 200, 200)


def test_hit_and_miss():
    assert reward("(150, 150)", BOX) == 1.0
    assert reward("(150,150)", BOX) == 1.0
    assert reward("(500, 500)", BOX) == 0.0
    assert reward("(100, 200)", BOX) == 1.0  # edges count


def test_special_tokens_and_prose():
    assert reward("<|box_start|>(150,150)<|box_end|>", BOX) == 1.0
    assert reward("Click at (150, 150).", BOX) == 1.0


def test_spam_is_rejected():
    assert parse_click("(150,150)(160,160)") is None
    assert reward("(150,150), (900,900), (120,120)", BOX) == 0.0


def test_boxes_and_truncated_boxes_are_rejected():
    assert parse_click("(120, 130, 180, 190)") is None   # a box is not a click
    assert parse_click("(733, 414, 886,") is None         # truncated at the token limit
    assert parse_click("(733, 414, 8") is None


def test_garbage_and_out_of_range():
    assert parse_click("") is None
    assert parse_click("no point here") is None
    assert parse_click("(1500, 150)") is None
    assert parse_click("(-5, 150)") is None


def test_synth_is_deterministic_and_valid():
    a, b = make_sample(5, 0.9, 0.9), make_sample(5, 0.9, 0.9)
    assert a.instruction == b.instruction and a.target == b.target and a.ring == b.ring
    assert a.image.size == (W, H)
    for seed in range(40):
        s = make_sample(seed)
        x0, y0, x1, y1 = s.box
        assert 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000
        labels = [e.label for e in s.elements]
        assert len(labels) == len(set(labels)), "labels must be unique so the instruction is unambiguous"


def test_cue_control():
    n = 150
    for name, (pp, pr), want_pos, want_ring in [("biased", SPLITS["biased"], 0.9, 0.9),
                                                 ("adv", SPLITS["adversarial_test"], 0.0, 0.0)]:
        ss = [make_sample(TEST_SEED_BASE + i, pp, pr) for i in range(n)]
        pos = sum(s.target_in_left for s in ss) / n
        ring = sum(s.ring == s.target for s in ss) / n
        assert abs(pos - want_pos) < 0.12, (name, pos)
        assert abs(ring - want_ring) < 0.12, (name, ring)
