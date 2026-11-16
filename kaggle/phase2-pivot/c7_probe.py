"""Weight Doctor Phase 2 — stronger-forcing follow-up (C7 probe).

Question: C6 (target +5 logit, alpha 0.0866) closes 5/9
easy problems but 0/9 hard ones — it does not close at
all on hard problems. The bias is too weak to overcome
the longer, more involved self-checking on multi-step
problems. The model derives the answer in thinking on
all 9 hard problems (first appearance at tokens
336-922) and never stops.

Does a stronger static bias close hard problems, and
at what cost on easy ones? A stronger bias should
close more — but it should also start closing easy
thinking too early, before derivation, and producing
wrong answers. The interesting result is the
tradeoff, not just the hard score. That is why the
decision rule below is two-sided.

Candidates (same method as C6, new alphas): one
lm_head row, alpha = target_logit_increase /
||mean_hidden||, mean hidden from Sara, derived
in-run exactly as Stage 2 did. The tied-embedding
disclosure applies to both, at larger magnitude.
- C7a: target +10 logit (expected alpha ~0.1731).
- C7b: target +20 logit (expected alpha ~0.3463).
Both are evaluated regardless of outcome (no stop
rule between them).

Explicitly out of this probe: a growing bias (a
logit bonus that increases with thinking length).
That is a decoding-time intervention, not a weight
edit, and belongs in its own plan if the static
tradeoff here turns out bad.

Batteries: HARD (9 multi-step problems, same as the
hard probe, frozen Nov 7 in C6_HARD_PROBE_PLAN.md),
TARGET (9 original target prompts), BREAKAGE (14
original breakage prompts). Baseline measured
in-run on all three. Gate: C6 (+5) re-run on TARGET
first; it must reproduce 5/9 or the run STOPS.

Decision rule, frozen before the run, per alpha.
Let H = close+correct on HARD, E = close+correct
on TARGET, R = breakage regressions (baseline
passes, candidate fails) against the in-run
baseline.
- EXTENDS iff H >= 5/9 AND E >= 5/9 AND R = 0.
- TRADEOFF iff H >= 5/9 but E < 5/9 or R > 0.
- LIMIT iff H < 5/9 for that alpha. If both alphas
  hit LIMIT, the forcing approach hits its limit.
Each alpha is judged on its own; a split between
them is itself the result.

First-appearance analysis (read-only, from saved
outputs, as in the hard probe): for the prompts
each alpha closes, did closure come before or after
derivation? Diagnostic only, not a kill bar.

One fix carried over from the hard probe: RAW
OUTPUTS ARE SAVED for every prompt (output +
generated token ids), so success flags can be
independently re-checked from text afterwards.

Pinned setup: Qwen/Qwen3.5-0.8B, float16, thinking
ON, temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token
cap, presence_penalty attempted with the per-call
fallback. Do NOT pip-upgrade transformers for this
run. Results are written after every step to
c7_probe_results.json; receipts to c7_probe_receipts/.
"""

import json
import re
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_TOKENS = 1024
SAMPLING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
}

# Frozen batteries — must match weight_doctor/phase2/batteries.py.
TARGET = [
    ("Maria has 24 apples. She gives 7 to her brother and buys 15 more. How many apples does she have now?", "32"),
    ("A baker makes 5 trays with 12 muffins each. He sells 38 muffins. How many muffins are left?", "22"),
    ("Tom reads 18 pages a day. How many pages does he read in 2 weeks?", "252"),
    ("A rectangle is 9 cm long and 4 cm wide. What is its area in square cm?", "36"),
    ("A train travels 60 km per hour for 3 hours, then travels 40 km more. How far did it travel in total, in km?", "220"),
    ("Jake earns $8 per hour and works 6 hours a day, 5 days a week. How much does he earn in a week, in dollars?", "240"),
    ("A store sells notebooks for $3 each. Ana buys 7 notebooks and pays with $25. How much change does she get, in dollars?", "4"),
    ("What is the capital of France? Answer in one sentence.", "Paris"),
    ("What colour is the sky on a clear day? Answer in one sentence.", "blue"),
]
BREAKAGE = [
    ("Sara has 3 boxes with 14 pencils each, plus 5 loose pencils. How many pencils does she have in total?", "47"),
    ("There are 48 students. Each table seats 6 students. How many tables are needed?", "8"),
    ("A garden has 5 rows with 11 plants each. 9 plants die. How many plants are left?", "46"),
    ("What is 2 + 3? Answer with just the number.", "5"),
    ("What is the capital of Japan? Answer in one sentence.", "Tokyo"),
    ("What is 7 + 8? Answer with just the number.", "15"),
    ("How many days are in a week? Answer with just the number.", "7"),
    ("What colour is grass on a sunny day? Answer in one sentence.", "green"),
    ("A box has 6 pens. Tom buys 4 boxes. How many pens does he have?", "24"),
    ("What is 10 - 4? Answer with just the number.", "6"),
    ("Name the planet we live on. Answer in one sentence.", "Earth"),
    ("Sara has 5 apples and eats 2. How many apples are left?", "3"),
    ("What is the first month of the year? Answer in one word.", "January"),
    ("A car travels 30 km in 1 hour. How far does it travel in 2 hours at the same speed, in km?", "60"),
]


# Hard battery — frozen Nov 7, 2026, in C6_HARD_PROBE_PLAN.md,
# before this run. Nine multi-step problems, single-number
# answers, verified by hand; no overlap with the original 23
# or the held-out 17; guard-checked (no repeat/count-again).
HARD = [
    ("A bakery bakes 240 loaves in the morning. They sell 3/8 of them before noon, then 145 in the afternoon. How many loaves are left at closing?", "5"),
    ("A train travels 120 km in 2 hours, then 180 km in 3 hours. What is its average speed in km per hour?", "60"),
    ("Sara has 4 shelves with 18 books each. She gives away 25 books and buys 12 more. How many books does she have now?", "59"),
    ("Tom earns $15 per hour. He works 6 hours a day for 5 days, then spends $120 on groceries and $45 on a ticket. How much money does he have left?", "285"),
    ("A farmer has 96 apples. He packs 1/4 of them into boxes, gives 18 to a neighbour, and the rest are split equally between 2 baskets. How many apples are in each basket?", "27"),
    ("A car drives 90 km in 1.5 hours, stops for 30 minutes, then drives 60 km in 1 hour. What is its average speed in km per hour over the whole trip, counting the stop?", "50"),
    ("A shop sells pens in packs of 8 for $6 and packs of 12 for $8. Maria needs 24 pens. What is the cheapest total cost in dollars?", "16"),
    ("Lily reads 15 pages on Monday, twice as many on Tuesday as Monday, and 10 fewer on Wednesday than Tuesday. How many pages does she read in total?", "65"),
    ("A farmer collects 5 dozen eggs each morning. He sells 3 dozen at the market and uses 14 eggs for baking. How many eggs are left?", "10"),
]

BANNED_PHRASES = [
    "repeat", "again, then again", "list them again",
    "count from", "one number per line", "50 times",
    "write it again", "then write it again",
]
MARKERS = ("wait", "okay", "final answer", "check", "confirm")
SCAN_PROMPTS = [TARGET[0], TARGET[7]]  # Maria, France (Stage 1 reps)


def check_guard(prompts):
    bad = [p for p in prompts if any(b in p.lower() for b in BANNED_PHRASES)]
    if bad:
        raise ValueError(f"prompt guard failed: {bad}")


def check_success(output, new_tokens, expected):
    closed = ("</think>" in output) and (new_tokens < MAX_TOKENS)
    if not closed:
        return False, False, False, "did not close thinking"
    after = output.split("</think>", 1)[1]
    if expected.isdigit():
        correct = re.search(rf"\b{re.escape(expected)}\b", after) is not None
    else:
        correct = expected.lower() in after.lower()
    if not correct:
        return False, True, False, "closed but wrong final answer"
    return True, True, True, "closed with correct final answer"


def marker_counts(output):
    low = output.lower()
    return {m: low.count(m) for m in MARKERS}


def build_text(tok, prompt, enable_thinking=True):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate(model, tok, prompt, device, max_tokens=MAX_TOKENS,
             enable_thinking=True, seed=0, return_sequences=False):
    torch.manual_seed(seed)
    text = build_text(tok, prompt, enable_thinking=enable_thinking)
    inputs = tok([text], return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    gen = dict(SAMPLING)
    pp_used = "presence_penalty" in gen
    if return_sequences:
        gen["return_dict_in_generate"] = True
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id, **gen)
    except (TypeError, ValueError) as e:
        if "presence_penalty" not in gen:
            raise
        print(f"presence_penalty rejected ({e}); retrying without it")
        del gen["presence_penalty"]
        pp_used = False
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id, **gen)
    elapsed = time.time() - t0
    if hasattr(out, "sequences"):
        seq = out.sequences[0]
    else:
        seq = out[0]
    new_tokens = seq[prompt_len:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    result = {
        "prompt": prompt,
        "output": decoded,
        "output_tail": decoded[-300:],
        "new_tokens": int(len(new_tokens)),
        "hit_token_cap": bool(len(new_tokens) >= max_tokens),
        "seconds": round(elapsed, 2),
        "presence_penalty_used": pp_used,
    }
    if return_sequences:
        result["generated_token_ids"] = [int(t) for t in new_tokens.tolist()]
        result["prompt_len"] = int(prompt_len)
    return result


def get_layers(model):
    for cand in ("model.layers", "language_model.layers",
                 "model.language_model.layers"):
        obj = model
        try:
            for part in cand.split("."):
                obj = getattr(obj, part)
            return obj, cand
        except AttributeError:
            continue
    raise AttributeError("no layers path found on model")


def get_mlp(layer):
    for cand in ("mlp", "feed_forward", "ffn"):
        if hasattr(layer, cand):
            return getattr(layer, cand), cand
    raise AttributeError("no MLP module found on layer")


def get_down_proj(mlp):
    for cand in ("down_proj", "fc2", "c_proj", "dense"):
        if hasattr(mlp, cand):
            return getattr(mlp, cand), cand
    raise AttributeError("no down_proj module found on MLP")


def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head, "model.lm_head"
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head, "get_output_embeddings()"
    raise AttributeError("no lm_head found on model")


def version(name):
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


def chunk_boundaries(total, n):
    return [round(i * total / n) for i in range(n + 1)]


def pick_best_scan(rows):
    """Frozen selection rule for scan rows.

    rows: one row per (group, prompt) with closed,
    marker_total, tokens. Returns (best_group, summary)
    where summary is per-group: both_closed, markers,
    tokens. Sort: both closed first, then fewest
    markers, then fewest tokens."""
    groups = {}
    for r in rows:
        g = groups.setdefault(r["group"], {"both_closed": 0, "markers": 0,
                                           "tokens": 0, "prompts": 0})
        g["both_closed"] += 1 if r["closed"] else 0
        g["markers"] += r["marker_total"]
        g["tokens"] += r["tokens"]
        g["prompts"] += 1
    ranked = sorted(groups.items(),
                    key=lambda kv: (-kv[1]["both_closed"],
                                    kv[1]["markers"], kv[1]["tokens"]))
    return ranked[0][0], {k: v for k, v in ranked}


def evaluate_verdict(target_success, baseline_breakage,
                     candidate_breakage, weights_changed,
                     total_weights, layer_limited):
    target_hits = sum(1 for s in target_success if s)
    regressions = [i for i, (b, c) in
                   enumerate(zip(baseline_breakage, candidate_breakage))
                   if b and not c]
    weight_fraction = (weights_changed / total_weights
                       if total_weights else None)
    p1 = target_hits >= 5 and len(regressions) == 0
    p2 = (weight_fraction is not None
          and weight_fraction <= 0.001 and bool(layer_limited))
    return {
        "target_hits": target_hits, "target_total": 9,
        "breakage_regressions": regressions,
        "breakage_regression_count": len(regressions),
        "weight_fraction": weight_fraction,
        "weights_changed": weights_changed,
        "layer_limited": bool(layer_limited),
        "kill_p1_pass": p1, "kill_p2_pass": p2,
        "passes_both": p1 and p2,
    }


def receipt_markdown(receipt):
    lines = [f"# Receipt — {receipt['candidate']}\n",
             f"Family: {receipt['family']}\n",
             f"Scope: {receipt['scope']}\n"]
    d = receipt["diff"]
    lines.append(f"Diff: {d.get('weights_changed')} weights changed "
                 f"({d.get('weight_fraction')}), "
                 f"L2={d.get('l2_norm')}, "
                 f"layer-limited={d.get('layer_limited')}, "
                 f"edit_type={d.get('edit_type')}\n")
    if d.get("note"):
        lines.append(f"Note: {d['note']}\n")
    t = receipt["target"]
    lines.append(f"Target: {t['hits']}/{t['total']}\n")
    for r in t["per_prompt"]:
        lines.append(f"- {'PASS' if r['success'] else 'fail'}: "
                     f"{r['prompt'][:70]} "
                     f"(closed={r['closed']}, correct={r['correct']}, "
                     f"tokens={r['new_tokens']})\n")
    b = receipt["breakage"]
    lines.append(f"\nBreakage regressions: {b['regression_count']} "
                 f"at indices {b['regressions']}\n")
    v = receipt["verdict"]
    lines.append(f"Kill P1: {v['kill_p1_pass']}, "
                 f"Kill P2: {v['kill_p2_pass']}\n")
    return "".join(lines)



def walk_tokens(tok, ids, expected):
    """Token-walk a generated id list.

    Returns (first_appearance_idx, close_idx): the index of the
    first generated token after which the cumulative decoded
    text contains the expected answer as a standalone number
    (None if never), and the index of the token where </think>
    first appears (None if never). Only tokens before the
    close token count for appearance — the final answer after
    </think> is not derivation in thinking."""
    cumulative = ""
    first = None
    close = None
    for i, tid in enumerate(ids):
        cumulative += tok.decode([tid], skip_special_tokens=False)
        if "</think>" in cumulative:
            close = i
            break
        if first is None and re.search(
                rf"\b{re.escape(expected)}\b", cumulative):
            first = i
    return first, close


def c7_receipt_markdown(receipt):
    lines = [f"# Receipt — {receipt['candidate']}\n",
             f"Scope: {receipt['scope']}\n"]
    d = receipt["diff"]
    lines.append(f"Diff: {d.get('weights_changed')} weights changed "
                 f"({d.get('weight_fraction')}), "
                 f"L2={d.get('l2_norm')}, "
                 f"layer-limited={d.get('layer_limited')}, "
                 f"edit_type={d.get('edit_type')}\n")
    if d.get("note"):
        lines.append(f"Note: {d['note']}\n")
    t = receipt["target"]
    lines.append(f"Hard target: {t['hits']}/{t['total']}\n")
    for r in t["per_prompt"]:
        lines.append(f"- {'PASS' if r['success'] else 'fail'}: "
                     f"{r['prompt'][:70]} "
                     f"(closed={r['closed']}, correct={r['correct']}, "
                     f"tokens={r['new_tokens']})\n")
    if "easy_target" in receipt:
        e = receipt["easy_target"]
        lines.append(f"\nEasy target: {e['hits']}/{e['total']}\n")
    if "breakage" in receipt:
        b = receipt["breakage"]
        lines.append(f"Breakage regressions: {b['regression_count']} "
                     f"at indices {b['regressions']}\n")
    v = receipt["verdict"]
    if isinstance(v, dict) and "H" in v:
        lines.append(f"\nVerdict: {v.get('outcome', v)} "
                     f"(H={v['H']}/9, E={v['E']}/9, "
                     f"R={v['R']})\n")
    else:
        lines.append(f"\nVerdict: {v.get('outcome', v)}\n")
    return "".join(lines)



def decide_alpha(H, E, R):
    """Frozen per-alpha decision rule (see module docstring)."""
    if H >= 5 and E >= 5 and R == 0:
        return "EXTENDS"
    if H >= 5:
        return "TRADEOFF"
    return "LIMIT"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = {
        "model_id": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "accelerate_version": version("accelerate"),
        "max_tokens": MAX_TOKENS,
    }
    print(json.dumps(info, indent=2))
    check_guard([p for p, _ in HARD] + [p for p, _ in TARGET]
                + [p for p, _ in BREAKAGE])
    results = {"environment": info, "candidates": [],
               "not_evaluated": []}
    receipts_dir = Path("c7_probe_receipts")
    receipts_dir.mkdir(exist_ok=True)

    def save():
        with open("c7_probe_results.json", "w") as f:
            json.dump(results, f, indent=2)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()
    total_weights = sum(p.numel() for p in model.parameters())
    results["total_weights"] = int(total_weights)
    layers, layers_path = get_layers(model)
    mlp15, mlp_path = get_mlp(layers[15])
    down15, down_path = get_down_proj(mlp15)
    results["layers_path"] = layers_path
    results["layer15_down_proj_shape"] = tuple(down15.weight.shape)
    save()

    # ---- Step 0: resolve the </think> token id (again, from a closure) ----
    sara_prompt, sara_expected = BREAKAGE[0]
    sara = generate(model, tok, sara_prompt, device,
                    return_sequences=True)
    _, sara_closed_flag, sara_correct_flag, _ = check_success(
        sara["output"], sara["new_tokens"], sara_expected)
    resolved_id = None
    cumulative = ""
    for tid in sara["generated_token_ids"]:
        cumulative += tok.decode([tid], skip_special_tokens=False)
        if "</think>" in cumulative:
            resolved_id = tid
            break
    results["token_resolution"] = {
        "prompt": sara_prompt, "closed": sara_closed_flag,
        "correct": sara_correct_flag,
        "new_tokens": sara["new_tokens"],
        "convert_tokens_to_ids": tok.convert_tokens_to_ids("</think>"),
        "encode": tok.encode("</think>", add_special_tokens=False),
        "resolved_id": resolved_id}
    print(f"[token] Sara closed={sara_closed_flag} correct={sara_correct_flag} "
          f"tokens={sara['new_tokens']} resolved={resolved_id}")
    save()
    if resolved_id is None:
        results["not_evaluated"].append(
            {"candidate": "C7a_plus10",
             "reason": "no resolved </think> token id from Step 0."})
        results["not_evaluated"].append(
            {"candidate": "C7b_plus20",
             "reason": "no resolved </think> token id from Step 0."})
        save()
        print("STOP: no </think> token id resolved.")
        return results

    def run_battery(prompt_list):
        rows = []
        for prompt, expected in prompt_list:
            r = generate(model, tok, prompt, device,
                         return_sequences=True)
            success, closed, correct, reason = check_success(
                r["output"], r["new_tokens"], expected)
            rows.append({"prompt": prompt, "expected": expected,
                         "success": success, "closed": closed,
                         "correct": correct, "new_tokens": r["new_tokens"],
                         "reason": reason, "output": r["output"],
                         "generated_token_ids": r["generated_token_ids"],
                         "seconds": r["seconds"]})
        return rows

    # ---- Step 1: baseline on all three batteries ----
    baseline_hard = run_battery(HARD)
    baseline_target = run_battery(TARGET)
    baseline_breakage = run_battery(BREAKAGE)
    results["baseline"] = {"hard": baseline_hard,
                           "target": baseline_target,
                           "breakage": baseline_breakage}
    base_h = sum(1 for r in baseline_hard if r["success"])
    base_e = sum(1 for r in baseline_target if r["success"])
    base_b = sum(1 for r in baseline_breakage if r["success"])
    print(f"Baseline: hard {base_h}/9, target {base_e}/9, "
          f"breakage {base_b}/14")
    save()
    baseline_breakage_flags = [r["success"] for r in baseline_breakage]

    # ---- Derivation: mean hidden from Sara (same method as Stage 2) ----
    lm_head, lm_path = get_lm_head(model)
    if resolved_id >= lm_head.weight.shape[0]:
        results["not_evaluated"].append(
            {"candidate": "C7a_plus10",
             "reason": f"resolved id {resolved_id} out of range for "
                       f"lm_head vocab {lm_head.weight.shape[0]}."})
        results["not_evaluated"].append(
            {"candidate": "C7b_plus20",
             "reason": f"resolved id {resolved_id} out of range for "
                       f"lm_head vocab {lm_head.weight.shape[0]}."})
        save()
        print("STOP: resolved id out of lm_head range.")
        return results
    try:
        captures = []

        def lm_pre_hook(module, inputs):
            captures.append(inputs[0][:, -1, :].detach().cpu())

        handle = lm_head.register_forward_pre_hook(lm_pre_hook)
        try:
            _ = generate(model, tok, sara_prompt, device)
        finally:
            handle.remove()
        if len(captures) < 2:
            raise ValueError(f"lm_head captures too few: {len(captures)}")
        mean_hidden = torch.stack(captures).float().mean(dim=0)
        unit = mean_hidden / torch.norm(mean_hidden)
        proj = float(torch.norm(mean_hidden).item())
        unit_dev = unit.to(device=device, dtype=lm_head.weight.dtype)
        orig_row = lm_head.weight.data[resolved_id].clone()
        tied = False
        try:
            emb = model.get_input_embeddings()
            tied = (emb is not None and emb.weight.data_ptr()
                    == lm_head.weight.data_ptr())
        except Exception:
            tied = False
        results["lm_head"] = {"path": lm_path,
                              "think_token_id": resolved_id,
                              "proj": proj, "tied": tied}
    except Exception as e:
        for name in ("C7a_plus10", "C7b_plus20"):
            results["not_evaluated"].append(
                {"candidate": name,
                 "reason": f"derivation failed: {type(e).__name__}: {e}"})
        print(f"[lm_head] FAILED: {e}")
        save()
        return results
    save()
    print(f"[lm_head] proj={proj:.2f} tied={tied} "
          f"(expected alphas: +10 -> {10.0 / proj:.4f}, "
          f"+20 -> {20.0 / proj:.4f})")

    def make_diff(target_logit):
        alpha = target_logit / proj if proj > 0 else 0.0
        return {"weights_changed": int(lm_head.weight.shape[1]),
                "weight_fraction": (lm_head.weight.shape[1]
                                    / total_weights),
                "l2_norm": round(float(alpha), 4),
                "layer_limited": True,
                "edit_type": "weight edit",
                "think_token_id": resolved_id,
                "alpha": float(alpha),
                "target_logit_increase": float(target_logit),
                "lm_head_tied_to_embeddings": tied,
                "note": ("Alpha targets a +{:.0f} logit increase on "
                         "average hidden alignment; actual "
                         "per-token increase varies. If lm_head "
                         "is tied, this row edit also changes "
                         "the input embedding for that token."
                         .format(target_logit))}, alpha

    def evaluate(name, target_logit):
        diff, alpha = make_diff(target_logit)

        def apply_fn(_a=alpha, _u=unit_dev):
            with torch.no_grad():
                lm_head.weight.data[resolved_id] = orig_row + _a * _u

        def restore_fn():
            with torch.no_grad():
                lm_head.weight.data[resolved_id] = orig_row
        apply_fn()
        try:
            hard_rows = run_battery(HARD)
            target_rows = run_battery(TARGET)
            breakage_rows = run_battery(BREAKAGE)
        finally:
            restore_fn()
        H = sum(1 for r in hard_rows if r["success"])
        E = sum(1 for r in target_rows if r["success"])
        R = [i for i, (b, c) in
             enumerate(zip(baseline_breakage_flags,
                           [r["success"] for r in breakage_rows]))
             if b and not c]
        verdict = decide_alpha(H, E, len(R))
        print(f"[{name}] hard {H}/9, target {E}/9, "
              f"regressions {len(R)} at {R} -> {verdict}")
        return {"hard_rows": hard_rows, "target_rows": target_rows,
                "breakage_rows": breakage_rows, "H": H, "E": E,
                "R": R, "verdict": verdict, "diff": diff}

    def write_receipt(name, family, outcome):
        receipt = {
            "candidate": name, "family": family,
            "scope": ("Qwen3.5-0.8B, thinking mode ON, Kaggle T4, "
                      "1024-token cap. Batteries: HARD (9 multi-step "
                      "problems, frozen Nov 7), TARGET (9 original "
                      "targets), BREAKAGE (14 original breakage "
                      "prompts). No claim about other models, modes, "
                      "or batteries."),
            "diff": outcome["diff"],
            "target": {"hits": outcome["H"], "total": 9,
                       "per_prompt": outcome["hard_rows"]},
            "easy_target": {"hits": outcome["E"], "total": 9,
                            "per_prompt": outcome["target_rows"]},
            "breakage": {"regressions": outcome["R"],
                         "regression_count": len(outcome["R"]),
                         "per_prompt": outcome["breakage_rows"]},
            "verdict": {"outcome": outcome["verdict"],
                        "H": outcome["H"], "E": outcome["E"],
                        "R": outcome["R"]},
            "environment": info,
            "sampling": SAMPLING,
        }
        with open(receipts_dir / f"{name}.json", "w") as f:
            json.dump(receipt, f, indent=2)
        with open(receipts_dir / f"{name}.md", "w") as f:
            f.write(c7_receipt_markdown(receipt))
        results["candidates"].append(receipt)
        save()
        return receipt

    def safe_evaluate(name, target_logit):
        try:
            return evaluate(name, target_logit)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            results.setdefault("candidate_errors", []).append(
                {"candidate": name, "error": err})
            results["not_evaluated"].append(
                {"candidate": name,
                 "reason": f"crashed during evaluation: {err}"})
            save()
            print(f"[{name}] CRASHED, continuing: {err}")
            return None

    # ---- Step 2: gate — C6 (+5) on TARGET must reproduce 5/9 ----
    gate = safe_evaluate("C6_gate", 5.0)
    if gate is None:
        print("STOP: gate evaluation crashed; cannot trust the run.")
        save()
        return results
    results["gate"] = {"H": gate["H"], "E": gate["E"],
                       "R": gate["R"], "verdict": gate["verdict"],
                       "target_hits": gate["E"]}
    write_receipt("C6_gate", "lm_head </think> bias (C6, +5, gate)",
                  gate)
    if gate["E"] != 5:
        results["setup_gate"] = {
            "passed": False, "control_hits": gate["E"],
            "reason": ("C6 scored != 5/9 on the original target "
                       "control — the setup changed. Run stopped; "
                       "no C7 numbers are reported.")}
        print(f"STOP: gate failed (C6 {gate['E']}/9 != 5/9). "
              "Setup changed.")
        save()
        return results
    results["setup_gate"] = {"passed": True, "control_hits": gate["E"]}
    print("[gate] passed: C6 reproduced 5/9 on TARGET")
    save()

    # ---- Steps 3-4: C7a (+10) and C7b (+20), both regardless ----
    outcomes = {}
    for name, target_logit in (("C7a_plus10", 10.0),
                               ("C7b_plus20", 20.0)):
        outcome = safe_evaluate(name, target_logit)
        if outcome is not None:
            outcomes[name] = outcome
            write_receipt(name, "lm_head </think> bias", outcome)
    save()

    # ---- Step 5: first-appearance analysis per alpha (read-only) ----
    analysis = {}
    for name, outcome in outcomes.items():
        rows = []
        for base_r, c7_r in zip(baseline_hard, outcome["hard_rows"]):
            first, _ = walk_tokens(tok, base_r["generated_token_ids"],
                                   base_r["expected"])
            _, c7_close = walk_tokens(tok, c7_r["generated_token_ids"],
                                      c7_r["expected"])
            if c7_close is None:
                reading = "did_not_close"
            elif first is None:
                reading = "no_derivation_in_baseline"
            elif c7_close < first:
                reading = "closed_before_derivation"
            else:
                reading = "closed_after_derivation"
            rows.append({"prompt": base_r["prompt"],
                         "expected": base_r["expected"],
                         "baseline_first_appearance_token": first,
                         "close_token": c7_close,
                         "success": c7_r["success"],
                         "reading": reading})
        analysis[name] = rows
    results["first_appearance"] = analysis
    save()

    # ---- Probe-level summary ----
    verdicts = {n: o["verdict"] for n, o in outcomes.items()}
    if verdicts and all(v == "LIMIT" for v in verdicts.values()):
        probe_outcome = ("LIMIT: neither alpha closed hard problems. "
                         "Static forcing hits its limit; the "
                         "growing-bias variant is the next thing "
                         "to scope.")
    elif verdicts:
        probe_outcome = ("Per-alpha verdicts: " + "; ".join(
            f"{n} {v} (H={outcomes[n]['H']}/9, "
            f"E={outcomes[n]['E']}/9, R={len(outcomes[n]['R'])})"
            for n, v in verdicts.items()) + ".")
    else:
        probe_outcome = "No alpha was evaluated."
    results["decision"] = {"verdicts": verdicts,
                           "outcome": probe_outcome}
    print(f"\nDecision: {probe_outcome}")

    with open("c7_probe_results.md", "w") as f:
        f.write("# Weight Doctor Phase 2 — stronger-forcing probe (C7)\\n\\n")
        f.write(f"Model: {MODEL_ID} on {info['gpu_name'] or device}\\n\\n")
        f.write(f"Baseline: hard {base_h}/9, target {base_e}/9, "
                f"breakage {base_b}/14\\n")
        f.write(f"Gate (C6 +5 on TARGET): {gate['E']}/9\\n")
        f.write(f"Token resolution: resolved </think> id = {resolved_id}\\n")
        f.write(f"lm_head proj: {proj:.2f}, tied: {tied}\\n\\n")
        for name, outcome in outcomes.items():
            f.write(f"- {name}: H={outcome['H']}/9, "
                    f"E={outcome['E']}/9, "
                    f"regressions {len(outcome['R'])} at "
                    f"{outcome['R']} -> {outcome['verdict']}\\n")
        f.write("\\nFirst-appearance analysis:\\n")
        for name, rows in analysis.items():
            f.write(f"{name}:\\n")
            for a in rows:
                f.write(f"- {a['reading']}: {a['prompt'][:60]} "
                        f"(first={a['baseline_first_appearance_token']}, "
                        f"close={a['close_token']}, "
                        f"success={a['success']})\\n")
        f.write(f"\\nDecision rule (frozen before the run): "
                f"{probe_outcome}\\n")
        f.write("\\nNot evaluated:\\n")
        for n in results["not_evaluated"]:
            f.write(f"- {n['candidate']}: {n['reason']}\\n")
        f.write("\\nReceipts in c7_probe_receipts/ (JSON + MD each). "
                "Raw outputs are saved in c7_probe_results.json.\\n")
    print("\\nDone.")
    return results


if __name__ == "__main__":
    main()
