"""Weight Doctor Phase 2 — C6 hard-problem probe.

Question: C6 "generalises" on easy arithmetic, where the
thinking is mostly wasted second-guessing and forcing
</think> early costs nothing. That claim is now public.
What was never tested: problems where the thinking does
real work. If the model needs the thinking to derive
the answer, C6's forced early closure should cut it off
before derivation and produce a wrong answer — and the
generalisation claim would need a narrower scope.

The crux is timing: on the easy battery the model
derives the answer quickly (Maria's 32, the baker's 22
appear early in thinking) and then second-guesses
forever, so C6 closing at ~600-700 tokens comes AFTER
derivation. On a multi-step problem derivation takes
longer. C6's lm_head bias raises the </think>
probability uniformly — it does not wait for the
answer. This probe measures whether closure comes
before or after derivation on hard problems.

Battery (frozen in C6_HARD_PROBE_PLAN.md before this
run): 9 multi-step problems with single-number
answers, verified by hand, no overlap with the
original 23 or the held-out 17. Controls: C6 on the
9 original target prompts in the same session (expect
~5/9, reproducing the frozen-battery result). If the
control scores below 3/9, the run STOPS — the setup
changed and the hard numbers would not be comparable.
The control is evaluated BEFORE the hard battery so
the stop rule can actually stop the run (the plan's
step list has hard before control; this reordering is
the only deviation, and it preserves the rule's
intent).

Diagnostic: thinking-OFF accuracy on the same 9 hard
problems (same method as Stage 1: stopped before the
cap AND the expected answer in the output). If
thinking-OFF is 0/9, the "hurts" comparison is moot
in that direction — the model cannot do them anyway —
and only the absolute bar carries weight.

Decision rule, frozen before the run. Let H = C6
close+correct on the hard battery, T = thinking-OFF
accuracy on the hard battery.
- Generalisation extends to hard problems iff H >= 5/9
  AND H >= T.
- Forcing closure hurts on hard problems iff H < T.
  The generalisation claim then narrows to easy
  problems, and the README must say so.
- Partial otherwise (T <= H < 5/9): C6 helps against
  non-termination but does not preserve full accuracy.

First-appearance analysis (read-only, from saved
outputs): in the unedited thinking traces, at what
token does the expected answer first appear, if at
all — and does C6's closure token come before or
after it? Diagnostic only, not a kill bar.

One fix over the generalisation probe: RAW OUTPUTS
ARE SAVED for every prompt (output + generated
token ids), so success flags can be independently
re-checked from text afterwards.

Pinned setup: Qwen/Qwen3.5-0.8B, float16, thinking
ON, temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token
cap, presence_penalty attempted with the per-call
fallback. C6 is the same alpha method as Stage 2
(mean hidden from Sara, target +5 logit,
tied-embedding status recorded). Do NOT pip-upgrade
transformers for this run. Results are written after
every step to hard_probe_results.json; receipts to
hard_probe_receipts/.
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


def hard_receipt_markdown(receipt):
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
    lines.append(f"Target: {t['hits']}/{t['total']}\n")
    for r in t["per_prompt"]:
        lines.append(f"- {'PASS' if r['success'] else 'fail'}: "
                     f"{r['prompt'][:70]} "
                     f"(closed={r['closed']}, correct={r['correct']}, "
                     f"tokens={r['new_tokens']})\n")
    v = receipt["verdict"]
    lines.append(f"\nVerdict: {v.get('outcome', v)}\n")
    return "".join(lines)


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
    check_guard([p for p, _ in HARD] + [p for p, _ in TARGET])
    results = {"environment": info, "candidates": [],
               "not_evaluated": []}
    receipts_dir = Path("hard_probe_receipts")
    receipts_dir.mkdir(exist_ok=True)

    def save():
        with open("hard_probe_results.json", "w") as f:
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
            {"candidate": "C6_lm_head_bias",
             "reason": "no resolved </think> token id from Step 0."})
        save()
        print("STOP: no </think> token id resolved.")
        return results

    # ---- Step 1: baseline, thinking ON, hard battery ----
    def run_battery(prompt_list, enable_thinking=True):
        rows = []
        for prompt, expected in prompt_list:
            r = generate(model, tok, prompt, device,
                         enable_thinking=enable_thinking,
                         return_sequences=True)
            if enable_thinking:
                success, closed, correct, reason = check_success(
                    r["output"], r["new_tokens"], expected)
                rows.append({"prompt": prompt, "expected": expected,
                             "success": success, "closed": closed,
                             "correct": correct, "new_tokens": r["new_tokens"],
                             "reason": reason, "output": r["output"],
                             "generated_token_ids": r["generated_token_ids"],
                             "seconds": r["seconds"]})
            else:
                stopped = r["new_tokens"] < MAX_TOKENS
                if expected.isdigit():
                    correct = re.search(
                        rf"\b{re.escape(expected)}\b", r["output"]) is not None
                else:
                    correct = expected.lower() in r["output"].lower()
                rows.append({"prompt": prompt, "expected": expected,
                             "stopped": stopped, "correct": correct,
                             "success": bool(stopped and correct),
                             "new_tokens": r["new_tokens"],
                             "output": r["output"],
                             "seconds": r["seconds"]})
        return rows

    baseline_hard = run_battery(HARD)
    results["baseline_hard"] = baseline_hard
    base_hits = sum(1 for r in baseline_hard if r["success"])
    print(f"Baseline (hard battery, thinking ON): {base_hits}/9")
    save()

    # ---- Step 2: thinking-OFF diagnostic on the hard battery ----
    off_hard = run_battery(HARD, enable_thinking=False)
    results["thinking_off_hard"] = off_hard
    off_hits = sum(1 for r in off_hard if r["success"])
    print(f"Thinking-OFF (hard battery): {off_hits}/9")
    save()

    # ---- Step 3: C6 lm_head bias (same method as Stage 2) ----
    lm_head, lm_path = get_lm_head(model)
    if resolved_id >= lm_head.weight.shape[0]:
        results["not_evaluated"].append(
            {"candidate": "C6_lm_head_bias",
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
        alpha = 5.0 / proj if proj > 0 else 0.0
        unit_dev = unit.to(device=device, dtype=lm_head.weight.dtype)
        orig_row = lm_head.weight.data[resolved_id].clone()
        tied = False
        try:
            emb = model.get_input_embeddings()
            tied = (emb is not None and emb.weight.data_ptr()
                    == lm_head.weight.data_ptr())
        except Exception:
            tied = False

        def apply_fn(_a=alpha, _u=unit_dev):
            with torch.no_grad():
                lm_head.weight.data[resolved_id] = orig_row + _a * _u

        def restore_fn():
            with torch.no_grad():
                lm_head.weight.data[resolved_id] = orig_row
        diff = {"weights_changed": int(lm_head.weight.shape[1]),
                "weight_fraction": (lm_head.weight.shape[1]
                                    / total_weights),
                "l2_norm": round(float(alpha), 4),
                "layer_limited": True,
                "edit_type": "weight edit",
                "think_token_id": resolved_id,
                "alpha": float(alpha),
                "target_logit_increase": 5.0,
                "lm_head_tied_to_embeddings": tied,
                "note": ("Alpha targets a +5 logit increase on "
                         "average hidden alignment; actual "
                         "per-token increase varies. If lm_head "
                         "is tied, this row edit also changes "
                         "the input embedding for that token.")}
        results["lm_head"] = {"path": lm_path,
                              "think_token_id": resolved_id,
                              "alpha": float(alpha), "tied": tied}
    except Exception as e:
        results["not_evaluated"].append(
            {"candidate": "C6_lm_head_bias",
             "reason": f"derivation failed: {type(e).__name__}: {e}"})
        print(f"[lm_head] FAILED: {e}")
        save()
        return results
    save()

    def evaluate_c6(name, prompt_list):
        apply_fn()
        try:
            rows = run_battery(prompt_list)
        finally:
            restore_fn()
        hits = sum(1 for r in rows if r["success"])
        print(f"[{name}] {hits}/{len(rows)}")
        return rows, hits

    # ---- Step 3a: control FIRST — C6 on the original 9 targets.
    # The plan's stop rule (control < 3/9 means the setup
    # changed) can only stop the run if the control runs
    # before the hard battery. ----
    control_rows, control_hits = evaluate_c6("C6_control", TARGET)
    results["c6_control"] = control_rows
    results["c6_control_hits"] = control_hits
    save()
    if control_hits < 3:
        results["setup_gate"] = {
            "passed": False,
            "control_hits": control_hits,
            "reason": ("C6 scored below 3/9 on the original target "
                       "control — the setup changed. Run stopped; "
                       "no hard-battery numbers are reported.")}
        print("STOP: control gate failed "
              f"({control_hits}/9 < 3/9). Setup changed.")
        save()
        return results
    results["setup_gate"] = {"passed": True,
                             "control_hits": control_hits}

    # ---- Step 3b: C6 on the hard battery ----
    hard_rows, hard_hits = evaluate_c6("C6_hard", HARD)
    results["c6_hard"] = hard_rows
    results["c6_hard_hits"] = hard_hits
    save()

    # ---- Step 4: first-appearance analysis (read-only) ----
    analysis = []
    for base_r, c6_r in zip(baseline_hard, hard_rows):
        first, _ = walk_tokens(tok, base_r["generated_token_ids"],
                               base_r["expected"])
        _, c6_close = walk_tokens(tok, c6_r["generated_token_ids"],
                                  c6_r["expected"])
        if c6_close is None:
            reading = "c6_did_not_close"
        elif first is None:
            reading = "no_derivation_in_baseline"
        elif c6_close < first:
            reading = "closed_before_derivation"
        else:
            reading = "closed_after_derivation"
        analysis.append({"prompt": base_r["prompt"],
                         "expected": base_r["expected"],
                         "baseline_first_appearance_token": first,
                         "c6_close_token": c6_close,
                         "c6_success": c6_r["success"],
                         "reading": reading})
    results["first_appearance"] = analysis
    save()

    # ---- Decision rule (frozen before the run) ----
    H, T = hard_hits, off_hits
    if H >= 5 and H >= T:
        outcome = ("GENERALISATION EXTENDS: C6 closes thinking on "
                   f"hard problems without losing accuracy "
                   f"(H={H}/9, thinking-off={T}/9).")
    elif H < T:
        outcome = ("FORCING CLOSURE HURTS: C6 scores below "
                   f"thinking-off on hard problems (H={H}/9, "
                   f"thinking-off={T}/9). The generalisation "
                   "claim narrows to easy problems.")
    else:
        outcome = ("PARTIAL: C6 helps against non-termination "
                   f"but does not preserve full accuracy on hard "
                   f"problems (H={H}/9, thinking-off={T}/9).")
    results["decision"] = {"H": H, "T": T, "control_hits": control_hits,
                           "outcome": outcome}
    print(f"\\nDecision: {outcome}")

    scope = ("Qwen3.5-0.8B, thinking mode ON, Kaggle T4, "
             "1024-token cap, HARD battery: 9 multi-step problems "
             "frozen Nov 7, 2026 in C6_HARD_PROBE_PLAN.md, no "
             "overlap with the original 23 or the held-out 17. "
             "Control: C6 on the original 9 target prompts. No "
             "claim about other models, modes, or batteries.")
    for name, rows, hits in (("C6_hard", hard_rows, hard_hits),
                             ("C6_control", control_rows, control_hits)):
        receipt = {
            "candidate": name, "family": "lm_head </think> bias",
            "scope": scope, "diff": diff,
            "target": {"hits": hits, "total": len(rows),
                       "per_prompt": rows},
            "verdict": {"outcome": outcome if name == "C6_hard"
                        else f"control, {hits}/9"},
            "environment": info,
            "sampling": SAMPLING,
        }
        with open(receipts_dir / f"{name}.json", "w") as f:
            json.dump(receipt, f, indent=2)
        with open(receipts_dir / f"{name}.md", "w") as f:
            f.write(hard_receipt_markdown(receipt))
        results["candidates"].append(receipt)
    save()

    with open("hard_probe_results.md", "w") as f:
        f.write("# Weight Doctor Phase 2 — C6 hard-problem probe\\n\\n")
        f.write(f"Model: {MODEL_ID} on {info['gpu_name'] or device}\\n\\n")
        f.write(f"Baseline (hard, thinking ON): {base_hits}/9\\n")
        f.write(f"Thinking-OFF (hard): {off_hits}/9\\n")
        f.write(f"C6 control (original targets): {control_hits}/9\\n")
        f.write(f"C6 hard: {hard_hits}/9\\n")
        f.write(f"Token resolution: resolved </think> id = {resolved_id}\\n\\n")
        f.write("First-appearance analysis:\\n")
        for a in analysis:
            f.write(f"- {a['reading']}: {a['prompt'][:60]} "
                    f"(first={a['baseline_first_appearance_token']}, "
                    f"close={a['c6_close_token']}, "
                    f"success={a['c6_success']})\\n")
        f.write(f"\\nDecision rule (frozen before the run): {outcome}\\n")
        f.write("\\nNot evaluated:\\n")
        for n in results["not_evaluated"]:
            f.write(f"- {n['candidate']}: {n['reason']}\\n")
        f.write("\\nReceipts in hard_probe_receipts/ (JSON + MD each). "
                "Raw outputs are saved in hard_probe_results.json.\\n")
    print("\\nDone.")
    return results


if __name__ == "__main__":
    main()
