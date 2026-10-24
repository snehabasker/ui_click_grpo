"""Minimal GRPO for single-turn UI grounding with a verifiable reward.

One optimiser step =
  for each of P fresh screenshots:
      sample G completions                      (model.generate, temperature 1)
      reward_i = 1[click inside target box]      (uiclick.reward)
      A_i      = (r_i - mean(r)) / (std(r) + eps)   group-relative advantage
      groups with std == 0 carry no signal and are skipped (their rate is logged)
      loss     = - sum_t A_i * logp_t / n_tokens  (+ beta * KL to the frozen base, optional)
  one AdamW step on the LoRA weights.

The policy is exactly on-policy (one update per rollout batch), so the PPO ratio is 1 and the clipped
surrogate reduces to the plain policy-gradient above.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .evaluate import test_samples, make_condition, predict, score
from .model import build_inputs, load_model, load_processor
from .meta import run_meta
from .reward import in_box, parse_click, reward as click_reward
from .synth import SPLITS, make_sample, norm_box


def _core(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _prefill(model, inp):
    """Run the (single) prompt once. Returns (last-position logits [1, V], KV cache, next text position)."""
    pos, _ = _core(model).model.get_rope_index(inp["input_ids"], image_grid_thw=inp["image_grid_thw"],
                                               attention_mask=inp["attention_mask"],
                                               mm_token_type_ids=inp["mm_token_type_ids"])
    out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], pixel_values=inp["pixel_values"],
                image_grid_thw=inp["image_grid_thw"], mm_token_type_ids=inp["mm_token_type_ids"],
                position_ids=pos, use_cache=True, logits_to_keep=1)
    return out.logits[:, -1].float(), out.past_key_values, pos.max() + 1


_bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # fp32 LoRA weights, bf16 activations (VRAM)


@torch.no_grad()
def sample_group(model, inp, G, max_new, eos, pad):
    """G temperature-1 samples for one prompt. The prompt (image + text) is processed once and its KV cache is
    shared by all samples, so a rollout costs one prefill + ~10 tiny decode steps instead of G prefills."""
    with _bf16():
        logits, cache, next_pos = _prefill(model, inp)
    cache.batch_repeat_interleave(G)
    logits = logits.expand(G, -1)
    attn = torch.ones(G, inp["input_ids"].shape[1], dtype=torch.long, device=logits.device)
    done = torch.zeros(G, dtype=torch.bool, device=logits.device)
    toks = []
    for t in range(max_new):
        tok = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
        tok = torch.where(done, torch.full_like(tok, pad), tok)
        toks.append(tok)
        done |= tok == eos
        if done.all():
            break
        attn = torch.cat([attn, torch.ones(G, 1, dtype=torch.long, device=attn.device)], 1)
        with _bf16():
            out = model(input_ids=tok[:, None], attention_mask=attn, past_key_values=cache, use_cache=True,
                        position_ids=(next_pos + t).view(1, 1, 1).expand(3, G, 1))
        logits = out.logits[:, -1].float()
    return torch.stack(toks, 1)


def completion_logps(model, inp, comp, comp_mask):
    """Exact log p(completion tokens | prompt), shape [G, Lc]. Prompt runs once WITH gradient; the G completions
    attend to its (graph-carrying) KV cache, so backward flows through the shared prompt a single time."""
    G, Lc = comp.shape
    with _bf16():
        first, cache, next_pos = _prefill(model, inp)
        cache.batch_repeat_interleave(G)
        attn = torch.cat([inp["attention_mask"].expand(G, -1), comp_mask], 1)
        pos = (next_pos + torch.arange(Lc, device=comp.device)).view(1, 1, Lc).expand(3, G, Lc)
        out = model(input_ids=comp, attention_mask=attn, past_key_values=cache, position_ids=pos, use_cache=True)
    logits = torch.cat([first.unsqueeze(0).expand(G, 1, -1), out.logits[:, :-1].float()], 1)
    return torch.gather(F.log_softmax(logits, -1), 2, comp.unsqueeze(-1)).squeeze(-1)


def train(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_seeds = [0, args.steps * args.prompts_per_step]
    (out / "run_config.json").write_text(json.dumps(run_meta({"args": vars(args), "train_seed_range": train_seeds,
                                                              "reward": f"1[click in box] + {args.format_bonus} * 1[parseable]; unparseable = 0"}), indent=2))
    torch.manual_seed(args.seed)

    processor = load_processor()
    model = load_model(trainable=True, lora_r=args.lora_r, lora_alpha=2 * args.lora_r)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    p_pos, p_ring = SPLITS[args.train_split]
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad = processor.tokenizer.pad_token_id
    G, P = args.group_size, args.prompts_per_step

    eval_sets = {}
    if args.eval_every:
        eval_sets = {s: test_samples(s, args.eval_n, "seen" if args.train_style == "seen" else "all")
                     for s in ["unbiased_test", "adversarial_test"]}
        if args.train_style == "seen":
            eval_sets["unbiased_heldout_style"] = test_samples("unbiased_test", args.eval_n, "heldout")
    log_f = open(out / "train_log.jsonl", "a")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        stats = dict(step=step, reward=[], informative=0, valid=[], length=[], decoy=[], left=[], unique=[], kl=[], loss=0.0)
        for j in range(P):
            s = make_sample((step - 1) * P + j, p_pos, p_ring, args.train_style)  # fresh screenshot every time, seeds < 1e7
            inp = build_inputs(processor, [s.image], [s.instruction])

            model.eval()
            comp = sample_group(model, inp, G, args.max_new_tokens, eos, pad)
            texts = processor.batch_decode(comp, skip_special_tokens=True)
            pts = [parse_click(t) for t in texts]
            hit = [p is not None and in_box(p, s.box) for p in pts]
            r = torch.tensor([click_reward(t, s.box, args.format_bonus) for t in texts], device=comp.device)

            stats["reward"].append(sum(hit) / G)  # logged as the pure hit rate, whatever the shaping
            stats["valid"].append(sum(p is not None for p in pts) / G)
            stats["unique"].append(len({t.strip() for t in texts}) / G)
            stats["length"].append((comp != pad).sum(1).float().mean().item())
            if s.ring != s.target:
                rb = norm_box(s.elements[s.ring].box)
                stats["decoy"].append(sum(p is not None and in_box(p, rb) for p in pts) / G)
            stats["left"].append(sum(p is not None and p[0] < 500 for p in pts) / max(1, sum(p is not None for p in pts)))
            if r.std() < 1e-6:
                continue  # all-hit or all-miss: zero advantage, no gradient
            stats["informative"] += 1
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)

            # mask: tokens up to and including the first <|im_end|>
            is_eos = (comp == eos).int()
            comp_mask = ((is_eos.cumsum(1) - is_eos) == 0).long()
            n_tok = comp_mask.sum().clamp(min=1)

            model.train()
            m = comp_mask.float()
            if args.beta > 0:
                with torch.no_grad(), model.disable_adapter():
                    ref_lp = completion_logps(model, inp, comp, comp_mask)
            lp = completion_logps(model, inp, comp, comp_mask)
            loss = -(adv.unsqueeze(1) * lp * m).sum()
            if args.beta > 0:
                d = ref_lp - lp
                kl = (torch.exp(d) - d - 1) * m
                loss = loss + args.beta * kl.sum()
                stats["kl"].append((kl.sum() / m.sum()).item())
            (loss / n_tok / P).backward()
            stats["loss"] += loss.item() / n_tok.item() / P

        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
        if stats["informative"]:
            opt.step()
        opt.zero_grad(set_to_none=True)

        mean = lambda v: float(sum(v) / len(v)) if v else None
        rec = dict(step=step, reward=mean(stats["reward"]), informative_frac=stats["informative"] / P,
                   valid_format=mean(stats["valid"]), unique_frac=mean(stats["unique"]), len=mean(stats["length"]),
                   decoy_click=mean(stats["decoy"]), left_click=mean(stats["left"]), kl=mean(stats["kl"]),
                   loss=stats["loss"], grad_norm=gn, peak_gb=torch.cuda.max_memory_allocated() / 1e9, elapsed=time.time() - t0)
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            gc.collect()
            torch.cuda.empty_cache()  # training leaves the allocator at the VRAM ceiling; without this eval is ~100x slower
            for name, ss in eval_sets.items():
                texts = predict(model, processor, *make_condition(ss, "real"), batch_size=args.eval_bs)
                sc = score(ss, texts)
                rec[f"eval_{name}_acc"] = sc["acc"]
                rec[f"eval_{name}_decoy"] = sc["decoy_rate"]
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items() if v is not None), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(out / "adapter")
    print("saved", out / "adapter")
