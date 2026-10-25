import pytest

from servebench.prompts import PROMPT, build_messages
from servebench.traces import TraceConfig, generate, load_trace


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    out = tmp_path_factory.mktemp("trace")
    cfg = TraceConfig(name="t", n_sessions=20, shared_fraction=0.75, n_shared_groups=3, warmup_sessions=2, seed=1)
    meta = generate(cfg, out)
    return out, cfg, meta, load_trace(out)


def test_prompt_matches_training_prompt():
    """Read PROMPT from the training module's source instead of importing it: importing uiclick.model pulls in torch and
    transformers, which makes the test fail on machines whose ML stack is unrelated to what is being checked."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "uiclick" / "model.py").read_text(encoding="utf-8")
    train_prompt = next(ast.literal_eval(n.value) for n in ast.parse(src).body
                        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "PROMPT" for t in n.targets))
    assert PROMPT == train_prompt


def test_turn_counts_and_shared_fraction_are_exact(trace):
    out, cfg, meta, sessions = trace
    assert len(sessions) == 20
    assert all(cfg.min_turns <= len(s["turns"]) <= cfg.max_turns for s in sessions)
    assert sum(s["shared"] for s in sessions) == 15
    assert meta["n_requests"] == sum(len(s["turns"]) for s in sessions)


def test_shared_sessions_have_identical_prefix_unique_do_not(trace):
    _, _, _, sessions = trace
    by_prefix = {}
    for s in sessions:
        by_prefix.setdefault(s["prefix_id"], []).append(s)
    for pid, group in by_prefix.items():
        sig = {(g["system"], tuple(e["sha256"] for e in g["fewshot"])) for g in group}
        if pid.startswith("shared"):
            assert len(sig) == 1, f"{pid}: sessions in one shared group must share the prefix byte for byte"
        else:
            assert len(group) == 1
    uniq = [s for s in sessions if not s["shared"]]
    assert len({s["system"] for s in uniq}) == len(uniq), "unique sessions must differ from the first characters"
    assert not {s["system"] for s in uniq} & {s["system"] for s in sessions if s["shared"]}


def test_generation_is_deterministic(trace, tmp_path):
    _, cfg, meta, _ = trace
    meta2 = generate(cfg, tmp_path)
    assert meta2["trace_sha256"] == meta["trace_sha256"]


def test_seeds_are_disjoint_from_train_and_test(trace):
    _, _, _, sessions = trace
    seeds = [t["seed"] for s in sessions for t in s["turns"]] + [e["seed"] for s in sessions for e in s["fewshot"]]
    assert all(1_000_000 < x < 10_000_000 or x >= 20_000_000 for x in seeds)
    assert len({t["seed"] for s in sessions for t in s["turns"]}) == sum(len(s["turns"]) for s in sessions)


def test_message_structure_per_history_mode(trace):
    out, _, _, sessions = trace
    s = dict(sessions[0], history="none")
    for hist, expect_img_turns in [("none", 0), ("text", 0), ("full", 2)]:
        sess = dict(s, history=hist)
        t = 2
        msgs = build_messages(sess, t, out)
        assert msgs[0]["role"] == "system"
        n_img = sum(1 for m in msgs if not isinstance(m["content"], str) for p in m["content"] if p["type"] == "image_url")
        assert n_img == len(sess["fewshot"]) + 1 + expect_img_turns
        assert msgs[-1]["role"] == "user" and msgs[-1]["content"][0]["type"] == "image_url"
        roles = [m["role"] for m in msgs]
        assert roles[1:1 + 2 * len(sess["fewshot"])] == ["user", "assistant"] * len(sess["fewshot"])


def test_replay_history_uses_ground_truth_answers(trace):
    out, _, _, sessions = trace
    s = sessions[0]
    msgs = build_messages(dict(s, history="text"), 2, out)
    answers = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert answers[-2:] == [s["turns"][0]["answer"], s["turns"][1]["answer"]]


def test_images_exist_and_hash_matches(trace):
    import hashlib
    out, _, _, sessions = trace
    for s in sessions[:3]:
        for t in s["turns"]:
            assert hashlib.sha256((out / t["image"]).read_bytes()).hexdigest() == t["sha256"]
