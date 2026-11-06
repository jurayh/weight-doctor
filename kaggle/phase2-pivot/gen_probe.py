"""Weight Doctor Phase 2 pivot — generalisation probe.

Question: does the Stage 2 pass generalise, or was it
a fit to the 9+14 battery? C6 lm_head bias passed
P1+P2 exactly at the bar (5/9, 0 regressions) on the
frozen battery; direction 0.5 was one regression away
(5/9, 1 regression). Both are re-tested here on a
HELD-OUT battery: 9 new target prompts + 8 new
breakage prompts, written and frozen BEFORE this run
(see NEW_TARGET / NEW_BREAKAGE below), never used in
candidate selection, never seen by any candidate.

Decision rule, frozen before the run: C6 generalises
if it scores >=5/9 target with zero regressions on
this held-out battery; otherwise the Phase 2 pass
does not generalise, and the verdict becomes "passes
the frozen battery, does not generalise". Direction
0.5 is a comparator, judged by the same bar. Both
candidates are evaluated regardless of outcome (no
stop rule in this probe), each on the FULL new
battery with its own receipt.

The candidates themselves are defined exactly as in
Stage 2 (same alpha method for C6, same direction
derivation from the original Maria prompt, same
float16 dtype fix). Only the battery is new.

Pinned setup: Qwen/Qwen3.5-0.8B, float16, thinking
ON, temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token
cap, presence_penalty attempted with the per-call
fallback. Do NOT pip-upgrade transformers for this
run. Results are written after every step to
gen_probe_results.json; receipts to gen_probe_receipts/.
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


# Held-out battery — frozen Nov 5, 2026, before this run.
# Same styles as the original batteries, all prompts and
# answers new. Arithmetic was checked by hand at writing
# time; the baseline pass rate is measured in the run,
# not assumed.
NEW_TARGET = [
    ("Emma has 31 marbles. She loses 9 and buys 12 more. How many marbles does she have now?", "34"),
    ("A farmer packs 6 boxes with 15 eggs each. He sells 47 eggs. How many eggs are left?", "43"),
    ("Lily reads 24 pages a day. How many pages does she read in 3 weeks?", "504"),
    ("A garden is 7 m long and 6 m wide. What is its area in square m?", "42"),
    ("A bus travels 45 km per hour for 2 hours, then travels 30 km more. How far did it travel in total, in km?", "120"),
    ("Mia earns $12 per hour and works 4 hours a day, 6 days a week. How much does she earn in a week, in dollars?", "288"),
    ("A store sells pencils for $2 each. Tom buys 9 pencils and pays with $20. How much change does he get, in dollars?", "2"),
    ("What is the capital of Italy? Answer in one sentence.", "Rome"),
    ("What colour is a ripe banana? Answer in one sentence.", "yellow"),
]
NEW_BREAKAGE = [
    ("Ben has 2 boxes with 18 stickers each, plus 4 loose stickers. How many stickers does he have in total?", "40"),
    ("There are 36 chairs. Each row seats 4 chairs. How many rows are needed?", "9"),
    ("A shelf has 4 rows with 13 books each. 7 books fall off. How many books are left?", "45"),
    ("What is 3 + 4? Answer with just the number.", "7"),
    ("What is the capital of Spain? Answer in one sentence.", "Madrid"),
    ("What is 9 + 6? Answer with just the number.", "15"),
    ("How many months are in a year? Answer with just the number.", "12"),
    ("What colour is snow on a sunny day? Answer in one sentence.", "white"),
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
    check_guard([p for p, _ in NEW_TARGET] + [p for p, _ in NEW_BREAKAGE])
    results = {"environment": info, "candidates": [],
               "not_evaluated": []}
    receipts_dir = Path("gen_probe_receipts")
    receipts_dir.mkdir(exist_ok=True)

    def save():
        with open("gen_probe_results.json", "w") as f:
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

    # ---- Step 1: baseline on the NEW battery ----
    def run_battery():
        t_rows, b_rows = [], []
        for prompt, expected in NEW_TARGET:
            r = generate(model, tok, prompt, device)
            success, closed, correct, reason = check_success(
                r["output"], r["new_tokens"], expected)
            t_rows.append({"prompt": prompt, "expected": expected,
                           "success": success, "closed": closed,
                           "correct": correct, "new_tokens": r["new_tokens"],
                           "reason": reason})
        for prompt, expected in NEW_BREAKAGE:
            r = generate(model, tok, prompt, device)
            success, closed, correct, reason = check_success(
                r["output"], r["new_tokens"], expected)
            b_rows.append({"prompt": prompt, "expected": expected,
                           "success": success, "closed": closed,
                           "correct": correct, "new_tokens": r["new_tokens"],
                           "reason": reason})
        return t_rows, b_rows

    baseline_t, baseline_b = run_battery()
    results["baseline"] = {"target": baseline_t, "breakage": baseline_b}
    target_hits = sum(1 for r in baseline_t if r["success"])
    break_hits = sum(1 for r in baseline_b if r["success"])
    results["baseline_summary"] = {"target": target_hits,
                                   "breakage": break_hits}
    print(f"Baseline (new battery): target {target_hits}/9, "
          f"breakage {break_hits}/8")
    save()
    baseline_breakage = [r["success"] for r in baseline_b]

    def evaluate(name, family, diff, apply_fn, restore_fn):
        apply_fn()
        try:
            t_rows, b_rows = run_battery()
        finally:
            restore_fn()
        verdict = evaluate_verdict(
            [r["success"] for r in t_rows], baseline_breakage,
            [r["success"] for r in b_rows], diff["weights_changed"],
            total_weights, diff["layer_limited"])
        receipt = {
            "candidate": name, "family": family,
            "scope": ("Qwen3.5-0.8B, thinking mode ON, Kaggle T4, "
                      "1024-token cap, HELD-OUT battery: these prompts "
                      "were written and frozen before this run and were "
                      "never used in candidate selection. No claim about "
                      "other models, modes, or batteries."),
            "diff": diff,
            "target": {"hits": verdict["target_hits"],
                       "total": verdict["target_total"],
                       "per_prompt": t_rows},
            "breakage": {"regressions": verdict["breakage_regressions"],
                         "regression_count": verdict["breakage_regression_count"],
                         "per_prompt": b_rows},
            "verdict": {"kill_p1_pass": verdict["kill_p1_pass"],
                        "kill_p2_pass": verdict["kill_p2_pass"],
                        "passes_both": verdict["passes_both"]},
            "environment": info,
            "sampling": SAMPLING,
        }
        with open(receipts_dir / f"{name}.json", "w") as f:
            json.dump(receipt, f, indent=2)
        with open(receipts_dir / f"{name}.md", "w") as f:
            f.write(receipt_markdown(receipt))
        results["candidates"].append(receipt)
        save()
        print(f"[{name}] target {verdict['target_hits']}/9, "
              f"regressions {verdict['breakage_regression_count']}, "
              f"P1={verdict['kill_p1_pass']} P2={verdict['kill_p2_pass']}")
        return verdict

    def safe_evaluate(name, family, diff, apply_fn, restore_fn):
        try:
            return evaluate(name, family, diff, apply_fn, restore_fn)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            results.setdefault("candidate_errors", []).append(
                {"candidate": name, "error": err})
            results["not_evaluated"].append(
                {"candidate": name,
                 "reason": f"crashed during evaluation: {err}"})
            save()
            print(f"[{name}] CRASHED, continuing: {err}")
            return {"passes_both": False, "kill_p1_pass": False,
                    "kill_p2_pass": False, "target_hits": 0,
                    "target_total": 9, "breakage_regressions": [],
                    "breakage_regression_count": 0, "error": err}

    # ---- Step 2: direction derivation (same method as Stage 2) ----
    direction = None
    direction_error = None
    try:
        captures = []

        def mlp_hook(module, inputs, output):
            captures.append(output[:, -1, :].detach().cpu())

        handle = mlp15.register_forward_hook(mlp_hook)
        try:
            maria = generate(model, tok, TARGET[0][0], device,
                             return_sequences=True)
        finally:
            handle.remove()
        gen_ids = maria["generated_token_ids"]
        if len(captures) != len(gen_ids):
            print(f"[direction] WARNING: {len(captures)} captures vs "
                  f"{len(gen_ids)} tokens; aligning on the first "
                  f"{min(len(captures), len(gen_ids))}")
        self_rows, other_rows = [], []
        for cap, tid in zip(captures, gen_ids):
            piece = tok.decode([tid], skip_special_tokens=True).lower()
            if "wait" in piece or "okay" in piece:
                self_rows.append(cap)
            else:
                other_rows.append(cap)
        if not self_rows or not other_rows:
            raise ValueError(f"direction classes empty: self={len(self_rows)} "
                             f"other={len(other_rows)}")
        self_mean = torch.stack(self_rows).float().mean(dim=0)
        other_mean = torch.stack(other_rows).float().mean(dim=0)
        model_dtype = down15.weight.dtype
        direction = (self_mean - other_mean).to(
            device=device, dtype=model_dtype)
        results["direction"] = {
            "self_check_tokens": len(self_rows),
            "other_tokens": len(other_rows),
            "direction_norm": float(torch.norm(direction).item()),
            "direction_dtype": str(model_dtype),
            "maria_tokens": maria["new_tokens"]}
        print(f"[direction] self={len(self_rows)} other={len(other_rows)} "
              f"norm={results['direction']['direction_norm']:.3f}")
    except Exception as e:
        direction_error = f"{type(e).__name__}: {e}"
        results["direction"] = {"error": direction_error}
        print(f"[direction] FAILED: {direction_error}")
    save()

    # ---- Step 3: C6 lm_head bias (same method as Stage 2) ----
    c6_verdict = None
    if resolved_id is None:
        results["not_evaluated"].append(
            {"candidate": "C6_lm_head_bias",
             "reason": "no resolved </think> token id from Step 0."})
    else:
        lm_head, lm_path = get_lm_head(model)
        if resolved_id >= lm_head.weight.shape[0]:
            results["not_evaluated"].append(
                {"candidate": "C6_lm_head_bias",
                 "reason": f"resolved id {resolved_id} out of range for "
                           f"lm_head vocab {lm_head.weight.shape[0]}."})
        else:
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
                c6_verdict = safe_evaluate("C6_lm_head_bias",
                                           "lm_head </think> bias",
                                           diff, apply_fn, restore_fn)
            except Exception as e:
                results["not_evaluated"].append(
                    {"candidate": "C6_lm_head_bias",
                     "reason": f"derivation failed: {type(e).__name__}: {e}"})
                print(f"[lm_head] FAILED: {e}")
    save()

    # ---- Step 4: direction 0.5 (the near-miss scale) ----
    if direction is not None:
        scale = 0.5
        handle_holder = {}

        def apply_fn(_s=scale):
            def hook(module, inputs, output):
                return (output - _s * direction).to(output.dtype)
            handle_holder["h"] = mlp15.register_forward_hook(hook)

        def restore_fn():
            if "h" in handle_holder:
                handle_holder["h"].remove()
        diff = {"weights_changed": 0, "weight_fraction": 0.0,
                "l2_norm": None, "layer_limited": True,
                "edit_type": "hook intervention (not a weight edit)",
                "scale": scale,
                "note": ("Hook changes no weights, so the P2 weight "
                         "count passes trivially. A P1 pass here still "
                         "needs a weight-edit conversion before claiming "
                         "a surgical weight fix.")}
        safe_evaluate("C_direction_0.5", "direction subtraction at L15",
                      diff, apply_fn, restore_fn)
    else:
        results["not_evaluated"].append(
            {"candidate": "C_direction_0.5",
             "reason": f"direction derivation failed: {direction_error}"})
    save()

    with open("gen_probe_results.md", "w") as f:
        f.write("# Weight Doctor Phase 2 — generalisation probe\\n\\n")
        f.write(f"Model: {MODEL_ID} on {info['gpu_name'] or device}\\n\\n")
        f.write(f"Held-out baseline: target {target_hits}/9, "
                f"breakage {break_hits}/8\\n")
        f.write(f"Token resolution: resolved </think> id = {resolved_id}\\n\\n")
        for c in results["candidates"]:
            v = c["verdict"]
            f.write(f"- {c['candidate']}: target {c['target']['hits']}/9, "
                    f"regressions {c['breakage']['regression_count']}, "
                    f"P1={v['kill_p1_pass']} P2={v['kill_p2_pass']}\\n")
        f.write("\\nDecision rule (frozen before the run): C6 generalises "
                "if it scores >=5/9 target with zero regressions on this "
                "held-out battery; otherwise the Phase 2 pass does not "
                "generalise. Direction 0.5 is a comparator, judged by "
                "the same bar.\\n")
        f.write("\\nNot evaluated:\\n")
        for n in results["not_evaluated"]:
            f.write(f"- {n['candidate']}: {n['reason']}\\n")
        f.write("\\nReceipts in gen_probe_receipts/ (JSON + MD each).\\n")
    print("\\nDone. Candidates:", len(results["candidates"]))
    return results


if __name__ == "__main__":
    main()
