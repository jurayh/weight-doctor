"""Weight Doctor Phase 2 pivot — Stage 2: candidate edits.

This is the BUILD, Stage 2, approved Nov 4, 2026, from
the Stage 1 diagnosis (results in kaggle/phase2-pivot/
results/STAGE1_VERDICT.md):

  Stage 1 facts it stands on (do not re-derive silently):
  - Baseline reproduced: target 0/9, known breakage 4/4.
  - New breakage 4/10, so breakage base rate is 8/14;
    breakage is a no-regression rule on the 8 passing.
  - Thinking-OFF 9/9: the failure is thinking-specific.
  - Layer-15 MLP ablation closes BOTH representative
    prompts (Maria 428 tok, France 101 tok, markers
    22->2 and 34->0); layers 0/12 close one each;
    6/19/23 close neither; Astra's 19 makes France
    worse (34->70 markers).
  - The Stage 1 </think> trajectory is INVALID
    (-inf logit with rank 2-7, recorded id 248069 vs
    closure top token 248044) and plays no part here.
    Step 0 of this notebook re-resolves the </think>
    token id from an actual closure before any
    lm_head candidate is trusted.

Frozen candidate order (from the Stage 1 verdict, in
order, at most 8 candidate evaluations, each on the
FULL 9+14 battery with its own receipt, never
cherry-picked on one prompt):
  C1: layer-15 column ablation, best subgroup
      (refined from the column scan).
  C2: layer-15 column ablation, best whole group
      (only if C1 does not pass both kills).
  C3-C5: direction subtraction at layer 15, scales
      0.25 / 0.5 / 1.0 (only if C1 and C2 fail).
  C6: lm_head </think> bias, target +5 logit
      (only if all previous fail, and only if Step 0
      resolves a valid token id).

Stop rule, frozen here: evaluation stops at the first
candidate passing BOTH Kill P1 and Kill P2. Later
families are then recorded as "not evaluated", never
as failures. A candidate that passes P1 but not P2
(or the reverse) does not stop the sequence.

Kill P1: >=5/9 target successes (close + correct)
AND zero breakage regressions vs THIS session's
baseline. Kill P2: <=0.1% of weights changed AND
layer/column/row-limited. L2 is reported, no bar.

Disclosure on edit types (read before judging P2):
- Column ablations (C1, C2) and the lm_head bias
  (C6) are REAL weight edits: the weight is changed,
  the battery is run, the weight is restored in a
  finally block. Their diff stats are exact.
- Direction subtraction (C3-C5) is a HOOK
  intervention: a vector is subtracted from layer-15
  MLP output at generation time. It changes no
  weights, so its weight-fraction check passes on
  count alone. A P1 pass there would still need a
  weight-edit conversion before anyone claims a
  surgical weight fix; the receipt says so.

Pinned setup: Qwen/Qwen3.5-0.8B, float16, thinking
ON, temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token
cap for scored prompts, presence_penalty attempted
with the per-call fallback (as in all probes).
Do NOT pip-upgrade transformers for this run.

Results are written after EVERY step to
phase2_stage2_results.json, so a session end loses
at most the step in progress. Receipts are written
per candidate to phase2_stage2_receipts/.
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
    check_guard([p for p, _ in TARGET] + [p for p, _ in BREAKAGE])
    results = {"environment": info, "candidates": [],
               "not_evaluated": []}
    receipts_dir = Path("phase2_stage2_receipts")
    receipts_dir.mkdir(exist_ok=True)

    def save():
        with open("phase2_stage2_results.json", "w") as f:
            json.dump(results, f, indent=2)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()
    total_weights = sum(p.numel() for p in model.parameters())
    results["total_weights"] = int(total_weights)
    layers, layers_path = get_layers(model)
    results["layers_path"] = layers_path
    results["num_layers"] = len(layers)
    mlp15, mlp_path = get_mlp(layers[15])
    down15, down_path = get_down_proj(mlp15)
    down_shape = tuple(down15.weight.shape)
    results["layer15_mlp_path"] = mlp_path
    results["layer15_down_proj_path"] = down_path
    results["layer15_down_proj_shape"] = down_shape
    print(f"Layers: {layers_path} x{len(layers)}; L15 MLP {mlp_path}, "
          f"down_proj {down_path} shape {down_shape}")
    save()

    # ---- Step 0: resolve the </think> token id from a closure ----
    # Stage 1's trajectory used 248069 and is invalid. Here
    # the id is found in an actual closed generation: run
    # the Sara prompt (closed at 666 tokens in every probe),
    # decode its generated ids with special tokens visible,
    # and walk tokens until the cumulative text contains
    # "</think>". That token's id is the resolved id.
    # If no such token is found, lm_head candidate C6 is
    # skipped (recorded, not failed).
    sara_prompt, sara_expected = BREAKAGE[0]
    sara = generate(model, tok, sara_prompt, device,
                    return_sequences=True)
    sara_success, sara_closed_flag, sara_correct_flag, _ = check_success(
        sara["output"], sara["new_tokens"], sara_expected)
    resolved_id = None
    cumulative = ""
    for tid in sara["generated_token_ids"]:
        piece = tok.decode([tid], skip_special_tokens=False)
        cumulative += piece
        if "</think>" in cumulative:
            resolved_id = tid
            break
    token_resolution = {
        "prompt": sara_prompt,
        "closed": sara_closed_flag,
        "correct": sara_correct_flag,
        "new_tokens": sara["new_tokens"],
        "convert_tokens_to_ids": tok.convert_tokens_to_ids("</think>"),
        "encode": tok.encode("</think>", add_special_tokens=False),
        "resolved_id": resolved_id,
        "resolved_method": ("token walk on a closed generation"
                            if resolved_id is not None else None),
    }
    results["token_resolution"] = token_resolution
    print(f"[token] Sara closed={sara_closed_flag} correct={sara_correct_flag} "
          f"tokens={sara['new_tokens']} convert={token_resolution['convert_tokens_to_ids']} "
          f"encode={token_resolution['encode']} resolved={resolved_id}")
    save()

    # ---- Step 1: baseline (this session) ----
    baseline = {"target": [], "breakage": []}
    for prompt, expected in TARGET:
        r = generate(model, tok, prompt, device)
        success, closed, correct, reason = check_success(
            r["output"], r["new_tokens"], expected)
        r.update({"expected": expected, "success": success,
                  "closed": closed, "correct": correct,
                  "reason": reason})
        baseline["target"].append(r)
        print(f"[baseline target] success={success} tokens={r['new_tokens']} "
              f"{prompt[:55]!r}")
    for prompt, expected in BREAKAGE:
        r = generate(model, tok, prompt, device)
        success, closed, correct, reason = check_success(
            r["output"], r["new_tokens"], expected)
        r.update({"expected": expected, "success": success,
                  "closed": closed, "correct": correct,
                  "reason": reason})
        baseline["breakage"].append(r)
        print(f"[baseline breakage] success={success} tokens={r['new_tokens']} "
              f"{prompt[:55]!r}")
    results["baseline"] = baseline
    target_hits = sum(1 for r in baseline["target"] if r["success"])
    known_hits = sum(1 for r in baseline["breakage"][:4] if r["success"])
    new_hits = sum(1 for r in baseline["breakage"][4:] if r["success"])
    results["baseline_summary"] = {
        "target": target_hits, "known_breakage": known_hits,
        "new_breakage": new_hits,
        "reproduced": target_hits == 0 and known_hits == 4}
    print(f"Baseline: target {target_hits}/9, known {known_hits}/4, "
          f"new {new_hits}/10")
    if not results["baseline_summary"]["reproduced"]:
        print("BASELINE DOES NOT REPRODUCE (expected 0/9 and 4/4). "
              "Candidates are still scored against THIS session's "
              "baseline, and the discrepancy is reported, not hidden.")
    save()
    baseline_breakage = [r["success"] for r in baseline["breakage"]]

    def run_battery():
        """Run the full 9+14 battery, return (target, breakage) rows."""
        t_rows, b_rows = [], []
        for prompt, expected in TARGET:
            r = generate(model, tok, prompt, device)
            success, closed, correct, reason = check_success(
                r["output"], r["new_tokens"], expected)
            t_rows.append({"prompt": prompt, "expected": expected,
                           "success": success, "closed": closed,
                           "correct": correct, "new_tokens": r["new_tokens"],
                           "reason": reason})
        for prompt, expected in BREAKAGE:
            r = generate(model, tok, prompt, device)
            success, closed, correct, reason = check_success(
                r["output"], r["new_tokens"], expected)
            b_rows.append({"prompt": prompt, "expected": expected,
                           "success": success, "closed": closed,
                           "correct": correct, "new_tokens": r["new_tokens"],
                           "reason": reason})
        return t_rows, b_rows

    def evaluate(name, family, diff, apply_fn, restore_fn):
        """Apply an edit, run the full battery, restore, receipt."""
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
                      "1024-token cap, this battery only. No claim "
                      "about other models, modes, or batteries."),
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
        """Candidate isolation (added Nov 5, 2026): a candidate
        that crashes must be recorded and skipped, not kill
        the whole run. The first rerun died at C_direction
        0.25 with a dtype RuntimeError, so C6 never ran and
        no final summary was written. evaluate() already
        restores the edit in a finally block, so a crash
        here leaves the model clean for the next candidate."""
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
            print(f"[{name}] CRASHED, continuing to next candidate: {err}")
            return {"passes_both": False, "kill_p1_pass": False,
                    "kill_p2_pass": False, "target_hits": 0,
                    "target_total": 9, "breakage_regressions": [],
                    "breakage_regression_count": 0, "error": err}

    # ---- Step 2: column scan inside layer-15 MLP ----
    intermediate = down_shape[1]
    hidden_dim = down_shape[0]
    bounds = chunk_boundaries(intermediate, 8)
    groups = [(bounds[i], bounds[i + 1]) for i in range(8)]
    results["column_scan_groups"] = groups
    print(f"[scan] intermediate={intermediate}, groups={groups}")

    def scan_groups(group_list, label):
        rows = []
        for gi, (start, end) in enumerate(group_list):
            def pre_hook(module, inputs, _s=start, _e=end):
                x = inputs[0].clone()
                x[..., _s:_e] = 0
                return (x,) + inputs[1:]
            handle = down15.register_forward_pre_hook(pre_hook)
            try:
                for prompt, expected in SCAN_PROMPTS:
                    r = generate(model, tok, prompt, device)
                    counts = marker_counts(r["output"])
                    rows.append({"scan": label, "group": gi,
                                 "start": start, "end": end,
                                 "prompt": prompt,
                                 "tokens": r["new_tokens"],
                                 "closed": ("</think>" in r["output"]
                                            and r["new_tokens"] < MAX_TOKENS),
                                 "marker_counts": counts,
                                 "marker_total": sum(counts.values())})
                    print(f"[scan {label} g{gi}] {prompt[:45]!r} "
                          f"closed={rows[-1]['closed']} tokens={r['new_tokens']} "
                          f"markers={rows[-1]['marker_total']}")
            finally:
                handle.remove()
        return rows

    scan_rows = scan_groups(groups, "group8")
    results["column_scan"] = scan_rows
    best_group, group_summary = pick_best_scan(scan_rows)
    results["column_scan_summary"] = group_summary
    print(f"[scan] best group: {best_group} {group_summary[best_group]}")
    save()

    # ---- Step 3: refine the best group into 4 subgroups ----
    g_start, g_end = groups[best_group]
    sub_bounds = chunk_boundaries(g_end - g_start, 4)
    subgroups = [(g_start + sub_bounds[i], g_start + sub_bounds[i + 1])
                 for i in range(4)]
    results["column_refine_subgroups"] = subgroups
    sub_rows = scan_groups(subgroups, "subgroup4")
    results["column_refine"] = sub_rows
    best_sub, sub_summary = pick_best_scan(sub_rows)
    results["column_refine_summary"] = sub_summary
    print(f"[refine] best subgroup: {best_sub} {sub_summary[best_sub]} "
          f"bounds={subgroups[best_sub]}")
    save()

    passed = False

    def zero_columns(start, end):
        orig = down15.weight.data[:, start:end].clone()
        l2 = float(torch.norm(down15.weight.data[:, start:end]).item())

        def apply_fn():
            with torch.no_grad():
                down15.weight.data[:, start:end] = 0

        def restore_fn():
            with torch.no_grad():
                down15.weight.data[:, start:end] = orig
        diff = {"weights_changed": int(hidden_dim * (end - start)),
                "weight_fraction": (hidden_dim * (end - start)
                                    / total_weights),
                "l2_norm": round(l2, 4),
                "layer_limited": True,
                "edit_type": "weight edit",
                "columns": [start, end],
                "note": None}
        return diff, apply_fn, restore_fn

    # ---- C1: best subgroup ablation ----
    s_start, s_end = subgroups[best_sub]
    diff, apply_fn, restore_fn = zero_columns(s_start, s_end)
    v = safe_evaluate("C1_subgroup_ablation",
                 "L15 column ablation (best subgroup)",
                 diff, apply_fn, restore_fn)
    passed = v["passes_both"]

    # ---- C2: best whole-group ablation (only if C1 fails) ----
    if not passed:
        diff, apply_fn, restore_fn = zero_columns(g_start, g_end)
        v = safe_evaluate("C2_group_ablation",
                     "L15 column ablation (best group)",
                     diff, apply_fn, restore_fn)
        passed = v["passes_both"]
    else:
        results["not_evaluated"].append(
            {"candidate": "C2_group_ablation",
             "reason": "C1 passed both kills; stop rule."})

    # ---- Direction derivation (only if C1/C2 fail) ----
    direction = None
    direction_error = None
    if not passed:
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
            # BUG FIX (Nov 4, 2026): the first Stage 2 run
            # proved captures align 1:1 with generated
            # tokens (1024 captures for 1024 generated
            # tokens), NOT prompt + one per token. The
            # prefill forward produces the first generated
            # token, so captures[i] is the hidden state
            # that predicts gen_ids[i]. The old code
            # assumed captures[1:] and failed with a
            # mismatch, skipping the direction candidates.
            # If lengths ever differ, align on the first
            # min(len) pairs and record a warning instead
            # of failing the whole derivation.
            gen_ids = maria["generated_token_ids"]
            if len(captures) != len(gen_ids):
                print(f"[direction] WARNING: {len(captures)} captures "
                      f"vs {len(gen_ids)} generated tokens; aligning "
                      f"on the first {min(len(captures), len(gen_ids))}")
            self_rows, other_rows = [], []
            for cap, tid in zip(captures, gen_ids):
                piece = tok.decode([tid], skip_special_tokens=True).lower()
                if "wait" in piece or "okay" in piece:
                    self_rows.append(cap)
                else:
                    other_rows.append(cap)
            if not self_rows or not other_rows:
                raise ValueError(
                    f"direction classes empty: self={len(self_rows)} "
                    f"other={len(other_rows)}")
            self_mean = torch.stack(self_rows).float().mean(dim=0)
            other_mean = torch.stack(other_rows).float().mean(dim=0)
            # DTYPE FIX (Nov 5, 2026): the first rerun crashed
            # here with "expected mat1 and mat2 to have the
            # same dtype, float != Half". The captures are
            # float32 (CPU, .float()), and .to(device) kept
            # float32, while the model is float16. Subtracting
            # a float32 direction from a float16 MLP output
            # promoted the output to float32, and the next
            # linear layer (Half weights) then failed. Cast
            # the direction to the model's weight dtype.
            model_dtype = down15.weight.dtype
            direction = (self_mean - other_mean).to(
                device=device, dtype=model_dtype)
            direction_error = None
            results["direction"] = {
                "self_check_tokens": len(self_rows),
                "other_tokens": len(other_rows),
                "direction_norm": float(torch.norm(direction).item()),
                "direction_dtype": str(model_dtype),
                "maria_tokens": maria["new_tokens"],
            }
            print(f"[direction] self={len(self_rows)} other={len(other_rows)} "
                  f"norm={results['direction']['direction_norm']:.3f}")
        except Exception as e:
            direction_error = f"{type(e).__name__}: {e}"
            results["direction"] = {"error": direction_error}
            print(f"[direction] FAILED: {direction_error}")
        save()

    # ---- C3-C5: direction subtraction, scales 0.25/0.5/1.0 ----
    if not passed and direction is not None:
        for scale in (0.25, 0.5, 1.0):
            if passed:
                results["not_evaluated"].append(
                    {"candidate": f"C_direction_{scale}",
                     "reason": "an earlier direction scale passed both kills; stop rule."})
                continue
            handle_holder = {}

            def apply_fn(_s=scale):
                def hook(module, inputs, output):
                    # Belt and braces on the dtype fix: the
                    # result must stay in the output's dtype
                    # (float16), even if the direction's dtype
                    # ever drifts again.
                    return (output - _s * direction).to(output.dtype)
                handle_holder["h"] = mlp15.register_forward_hook(hook)

            def restore_fn():
                if "h" in handle_holder:
                    handle_holder["h"].remove()
            diff = {"weights_changed": 0,
                    "weight_fraction": 0.0,
                    "l2_norm": None,
                    "layer_limited": True,
                    "edit_type": "hook intervention (not a weight edit)",
                    "scale": scale,
                    "note": ("Hook changes no weights, so the P2 "
                             "weight count passes trivially. A P1 pass "
                             "here still needs a weight-edit conversion "
                             "before claiming a surgical weight fix.")}
            v = safe_evaluate(f"C_direction_{scale}",
                         "direction subtraction at L15",
                         diff, apply_fn, restore_fn)
            passed = v["passes_both"]
    elif not passed and direction is None:
        results["not_evaluated"].append(
            {"candidate": "C3-C5_direction",
             "reason": f"direction derivation failed: {direction_error}"})

    # ---- C6: lm_head bias (only if all previous fail) ----
    if not passed:
        if resolved_id is None:
            results["not_evaluated"].append(
                {"candidate": "C6_lm_head_bias",
                 "reason": "no resolved </think> token id from Step 0."})
        else:
            lm_head, lm_path = get_lm_head(model)
            if resolved_id >= lm_head.weight.shape[0]:
                results["not_evaluated"].append(
                    {"candidate": "C6_lm_head_bias",
                     "reason": (f"resolved id {resolved_id} out of range "
                                f"for lm_head vocab {lm_head.weight.shape[0]}.")})
            else:
                try:
                    captures = []

                    def lm_pre_hook(module, inputs):
                        captures.append(inputs[0][:, -1, :].detach().cpu())

                    handle = lm_head.register_forward_pre_hook(lm_pre_hook)
                    try:
                        sara2 = generate(model, tok, sara_prompt, device)
                    finally:
                        handle.remove()
                    if len(captures) < 2:
                        raise ValueError(
                            f"lm_head captures too few: {len(captures)}")
                    # Same 1:1 alignment as the direction
                    # captures (see bug fix above): use all
                    # captures, not captures[1:]. The first
                    # run used captures[1:] and still passed;
                    # the mean over one extra token is a
                    # negligible change, recorded here so the
                    # rerun's alpha is comparable, not identical.
                    mean_hidden = torch.stack(captures).float().mean(dim=0)
                    unit = mean_hidden / torch.norm(mean_hidden)
                    proj = float(torch.norm(mean_hidden).item())
                    alpha = 5.0 / proj if proj > 0 else 0.0
                    # Same dtype fix as the direction: the row
                    # edit must stay in the lm_head weight
                    # dtype (float16), not float32.
                    unit_dev = unit.to(device=device,
                                       dtype=lm_head.weight.dtype)
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
                            lm_head.weight.data[resolved_id] = (
                                orig_row + _a * _u)

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
                            "note": ("Alpha targets a +5 logit increase "
                                     "on average hidden alignment; actual "
                                     "per-token increase varies. If lm_head "
                                     "is tied, this row edit also changes "
                                     "the input embedding for that token.")}
                    results["lm_head"] = {"path": lm_path,
                                          "think_token_id": resolved_id,
                                          "alpha": float(alpha),
                                          "tied": tied}
                    v = safe_evaluate("C6_lm_head_bias",
                                 "lm_head </think> bias",
                                 diff, apply_fn, restore_fn)
                    passed = v["passes_both"]
                except Exception as e:
                    results["not_evaluated"].append(
                        {"candidate": "C6_lm_head_bias",
                         "reason": f"derivation failed: {type(e).__name__}: {e}"})
                    print(f"[lm_head] FAILED: {e}")

    results["overall_pass"] = passed
    results["candidates_evaluated"] = len(results["candidates"])
    save()

    with open("phase2_stage2_results.md", "w") as f:
        f.write("# Weight Doctor Phase 2 — Stage 2 (candidate edits)\n\n")
        f.write(f"Model: {MODEL_ID} on {info['gpu_name'] or device}\n\n")
        f.write(f"Baseline: target {target_hits}/9, known breakage "
                f"{known_hits}/4, new breakage {new_hits}/10\n")
        f.write(f"Token resolution: resolved </think> id = {resolved_id} "
                f"(convert={token_resolution['convert_tokens_to_ids']}, "
                f"encode={token_resolution['encode']})\n\n")
        f.write(f"Candidates evaluated: {len(results['candidates'])}, "
                f"overall pass (P1+P2): {passed}\n\n")
        for c in results["candidates"]:
            v = c["verdict"]
            f.write(f"- {c['candidate']}: target {c['target']['hits']}/9, "
                    f"regressions {c['breakage']['regression_count']}, "
                    f"P1={v['kill_p1_pass']} P2={v['kill_p2_pass']}\n")
        f.write("\nNot evaluated:\n")
        for n in results["not_evaluated"]:
            f.write(f"- {n['candidate']}: {n['reason']}\n")
        f.write("\nReceipts in phase2_stage2_receipts/ (JSON + MD each).\n")
        f.write("No claim is made beyond this model, mode, sampling, "
                "and battery (Kill P4).\n")
    print(f"\nDone. Candidates: {len(results['candidates'])}, "
          f"overall pass: {passed}")
    return results


if __name__ == "__main__":
    main()
