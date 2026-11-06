"""Weight Doctor Phase 2 pivot — Stage 1: baseline + diagnosis.

This is the BUILD, not the plan. It runs, on a Kaggle T4,
the first two steps of PHASE2_PIVOT_PLAN.md:

  Step 1 — baseline re-run: target (9) + breakage (14)
    prompts at 1024 tokens, thinking ON, same sampling
    and seed as the probes. Must reproduce 0/9 target
    and 4/4 known breakage, or the build stops.
  Step 2 — diagnosis (read-only, no weight edit):
    (a) self-check marker rates from baseline outputs,
    (b) </think> logit trajectory on 3 prompts,
    (c) thinking-OFF diagnostic on the 9 target prompts,
    (d) layer ablation scan: 6 layers x 2 prompts,
        hooks only, weights never changed.

Candidate edits (Stage 2) are NOT run here: their layers
and scales must come from this diagnosis, not from a
guess. Stage 2 will be a separate notebook, built from
these results, with its candidate list frozen then.

Pinned setup: Qwen/Qwen3.5-0.8B, float16, thinking ON,
temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token cap
for scored prompts. presence_penalty is attempted with
the same per-call fallback as the probes (Kaggle
transformers rejects it with ValueError; omission is
recorded). Do NOT pip-upgrade transformers for this
run: comparability with the probes matters more than
the newest version, and any version change is logged
in the results.

Deviation from the plan, stated here, not hidden:
the plan listed "activation correlation + targeted
ablation" for the layer scan. Stage 1 does ablation
only. Ablation directly measures causal effect (does
removing this layer's MLP output change self-check
rate?); correlation is observational and would not
decide a candidate layer on its own. If ablation
identifies no layer, correlation can be added in
Stage 2 — it is not silently dropped, it is deferred
with a reason.

Results are written after EVERY step
(phase2_stage1_results.json), so a session end cannot
lose completed steps. The .md summary is written at
the end.

Troubleshooting: if you get the accelerate ImportError,
add a cell at the top with
  !pip install -q -U accelerate transformers
then Session -> Restart, then Run All again, and note
the version change when the results are judged.
"""

import json
import time
from collections import Counter
from importlib import metadata

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
ABLATION_LAYERS = [0, 6, 12, 15, 19, 23]


def check_guard(prompts):
    bad = [p for p in prompts if any(b in p.lower() for b in BANNED_PHRASES)]
    if bad:
        raise ValueError(f"prompt guard failed: {bad}")


def check_success(output, new_tokens, expected):
    import re
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
             enable_thinking=True, seed=0, output_scores=False):
    torch.manual_seed(seed)
    text = build_text(tok, prompt, enable_thinking=enable_thinking)
    inputs = tok([text], return_tensors="pt").to(device)
    gen = dict(SAMPLING)
    pp_used = "presence_penalty" in gen
    # BUG FIX (Nov 4, 2026): transformers generate() returns a
    # plain Tensor unless return_dict_in_generate=True, even
    # with output_scores=True. The first Stage 1 version
    # assumed output_scores implied a GenerateOutput and
    # crashed at out.sequences with AttributeError. Pass
    # return_dict_in_generate=True whenever scores are asked
    # for, and still handle a plain-Tensor return defensively
    # (scores=None, trajectory empty for that prompt) instead
    # of crashing the whole run.
    if output_scores:
        gen["return_dict_in_generate"] = True
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id,
                output_scores=output_scores, **gen)
    except (TypeError, ValueError) as e:
        if "presence_penalty" not in gen:
            raise
        print(f"presence_penalty rejected ({e}); retrying without it")
        del gen["presence_penalty"]
        pp_used = False
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id,
                output_scores=output_scores, **gen)
    elapsed = time.time() - t0
    if hasattr(out, "sequences"):
        seq = out.sequences[0]
        scores = out.scores if output_scores else None
    else:
        # Plain-Tensor return (e.g. a generate wrapper that
        # ignores return_dict_in_generate): scores unavailable.
        seq = out[0]
        scores = None
        if output_scores:
            print("WARNING: generate() returned a Tensor, not a "
                  "GenerateOutput; </think> trajectory unavailable "
                  "for this prompt, continuing without it.")
    new_tokens = seq[inputs["input_ids"].shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "output": decoded,
        "output_tail": decoded[-300:],
        "new_tokens": int(len(new_tokens)),
        "hit_token_cap": bool(len(new_tokens) >= max_tokens),
        "seconds": round(elapsed, 2),
        "presence_penalty_used": pp_used,
    }, scores


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


def version(name):
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


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

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()
    results = {"environment": info, "baseline": {}, "diagnosis": {}}

    def save():
        with open("phase2_stage1_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ---- Step 1: baseline ----
    baseline = {"target": [], "breakage": []}
    for prompt, expected in TARGET:
        r, _ = generate(model, tok, prompt, device)
        success, closed, correct, reason = check_success(
            r["output"], r["new_tokens"], expected)
        r.update({"expected": expected, "success": success,
                  "closed": closed, "correct": correct,
                  "reason": reason})
        baseline["target"].append(r)
        print(f"[baseline target] success={success} closed={closed} "
              f"tokens={r['new_tokens']} {r['seconds']}s {prompt[:55]!r}")
    for prompt, expected in BREAKAGE:
        r, _ = generate(model, tok, prompt, device)
        success, closed, correct, reason = check_success(
            r["output"], r["new_tokens"], expected)
        r.update({"expected": expected, "success": success,
                  "closed": closed, "correct": correct,
                  "reason": reason})
        baseline["breakage"].append(r)
        print(f"[baseline breakage] success={success} closed={closed} "
              f"tokens={r['new_tokens']} {r['seconds']}s {prompt[:55]!r}")
    results["baseline"] = baseline
    save()
    target_hits = sum(1 for r in baseline["target"] if r["success"])
    known_hits = sum(1 for r in baseline["breakage"][:4] if r["success"])
    print(f"Baseline: target {target_hits}/9, known breakage {known_hits}/4")
    if target_hits != 0 or known_hits != 4:
        print("BASELINE DOES NOT REPRODUCE (expected 0/9 and 4/4). "
              "Diagnosis below is still recorded, but candidate "
              "edits must NOT be evaluated against this baseline "
              "until the discrepancy is explained.")
        results["baseline_reproduced"] = False
    else:
        results["baseline_reproduced"] = True
    save()

    # ---- Step 2a: marker rates (from baseline outputs) ----
    # No re-generation: the two representative prompts'
    # baseline outputs are analysed here, so marker rates
    # describe exactly the generations already scored.
    diagnosis = {}
    marker_rows = []
    for r in (baseline["target"][0], baseline["target"][7]):
        low = r["output"].lower()
        counts = {m: low.count(m) for m in MARKERS}
        words = r["output"].split()
        blocks = []
        for i in range(0, len(words), 256):
            chunk = " ".join(words[i:i + 256]).lower()
            blocks.append(sum(chunk.count(m) for m in MARKERS))
        marker_rows.append({"prompt": r["prompt"], "tokens": r["new_tokens"],
                            "closed": r["closed"],
                            "marker_counts": counts,
                            "markers_per_256_block": blocks})
        print(f"[markers] {r['prompt'][:55]!r} tokens={r['new_tokens']} "
              f"counts={counts} blocks={blocks}")
    diagnosis["marker_rates"] = marker_rows
    results["diagnosis"] = diagnosis
    save()

    # ---- Step 2b: </think> logit trajectory (3 prompts) ----
    # Wrapped so a trajectory failure can never kill the
    # rest of Stage 1 again: thinking-OFF and ablation are
    # independent diagnosis steps and must still run. A
    # failed trajectory is recorded as an error, not as
    # evidence about the model.
    traj_rows = []
    traj_error = None
    try:
        think_id = tok.convert_tokens_to_ids("</think>")
        if think_id is None:
            enc = tok.encode("</think>", add_special_tokens=False)
            think_id = enc[0] if len(enc) == 1 else None
        print(f"</think> token id: {think_id}")
        traj_prompts = [TARGET[0], TARGET[7], BREAKAGE[0]]
        for prompt, expected in traj_prompts:
            r, scores = generate(model, tok, prompt, device,
                                 output_scores=True)
            points = []
            if scores is not None and think_id is not None:
                for step in range(0, len(scores), 32):
                    logits = scores[step][0]
                    think_logit = float(logits[think_id])
                    rank = int((logits > think_logit).sum()) + 1
                    top = int(torch.argmax(logits))
                    points.append({"step": step, "think_rank": rank,
                                   "think_logit": round(think_logit, 3),
                                   "top_token_id": top})
                # Always include the final step.
                if (len(scores) - 1) % 32 != 0:
                    logits = scores[-1][0]
                    think_logit = float(logits[think_id])
                    points.append({"step": len(scores) - 1,
                                   "think_rank": int((logits > think_logit).sum()) + 1,
                                   "think_logit": round(think_logit, 3),
                                   "top_token_id": int(torch.argmax(logits))})
            traj_rows.append({"prompt": prompt, "tokens": r["new_tokens"],
                              "think_token_id": think_id,
                              "trajectory": points})
            print(f"[trajectory] {prompt[:55]!r} tokens={r['new_tokens']} "
                  f"points={len(points)}")
    except Exception as e:
        traj_error = f"{type(e).__name__}: {e}"
        print(f"[trajectory] FAILED, continuing to thinking-OFF "
              f"and ablation: {traj_error}")
    diagnosis["think_trajectory"] = traj_rows
    diagnosis["think_trajectory_error"] = traj_error
    results["diagnosis"] = diagnosis
    save()

    # ---- Step 2c: thinking-OFF diagnostic ----
    off_rows = []
    for prompt, expected in TARGET:
        r, _ = generate(model, tok, prompt, device, enable_thinking=False)
        # No </think> expected when thinking is off: success =
        # stopped before cap AND expected answer in the output.
        stopped = r["new_tokens"] < MAX_TOKENS
        import re
        if expected.isdigit():
            correct = re.search(rf"\b{re.escape(expected)}\b", r["output"]) is not None
        else:
            correct = expected.lower() in r["output"].lower()
        off_rows.append({"prompt": prompt, "expected": expected,
                         "stopped": stopped, "correct": correct,
                         "success": bool(stopped and correct),
                         "new_tokens": r["new_tokens"],
                         "seconds": r["seconds"]})
        print(f"[thinking-off] success={stopped and correct} "
              f"tokens={r['new_tokens']} {prompt[:55]!r}")
    diagnosis["thinking_off"] = off_rows
    results["diagnosis"] = diagnosis
    save()

    # ---- Step 2d: layer ablation scan (hooks only) ----
    layers, layers_path = get_layers(model)
    print(f"Layers path: {layers_path}, {len(layers)} layers")
    ablation_prompts = [TARGET[0], TARGET[7]]
    ablation_rows = []
    for layer_idx in ABLATION_LAYERS:
        mlp = layers[layer_idx].mlp

        def zero_mlp(module, inputs, output, _mlp=mlp):
            return torch.zeros_like(output)

        handle = mlp.register_forward_hook(zero_mlp)
        try:
            for prompt, expected in ablation_prompts:
                r, _ = generate(model, tok, prompt, device)
                low = r["output"].lower()
                counts = {m: low.count(m) for m in MARKERS}
                ablation_rows.append({
                    "layer": layer_idx, "prompt": prompt,
                    "tokens": r["new_tokens"],
                    "closed": ("</think>" in r["output"]
                               and r["new_tokens"] < MAX_TOKENS),
                    "marker_counts": counts,
                    "marker_total": sum(counts.values())})
                print(f"[ablation L{layer_idx}] {prompt[:45]!r} "
                      f"tokens={r['new_tokens']} markers={sum(counts.values())}")
        finally:
            handle.remove()
    diagnosis["layer_ablation"] = ablation_rows
    diagnosis["layers_path"] = layers_path
    diagnosis["num_layers"] = len(layers)
    results["diagnosis"] = diagnosis
    save()

    with open("phase2_stage1_results.md", "w") as f:
        f.write("# Weight Doctor Phase 2 — Stage 1 (baseline + diagnosis)\n\n")
        f.write(f"Model: {MODEL_ID} on {info['gpu_name'] or device}\n\n")
        f.write(f"Baseline reproduced (0/9 target, 4/4 known breakage): "
                f"{results['baseline_reproduced']}\n\n")
        f.write(f"Baseline target: {target_hits}/9\n")
        f.write(f"Baseline known breakage: {known_hits}/4\n")
        f.write(f"Baseline new breakage: "
                f"{sum(1 for r in baseline['breakage'][4:] if r['success'])}/10\n\n")
        f.write("Diagnosis: marker rates, </think> trajectory, "
                "thinking-OFF, and layer ablation are in the JSON.\n\n")
        f.write("No weight was edited in Stage 1. Ablation used "
                "forward hooks only; hooks were removed after each "
                "layer, and model weights were never changed.\n")
    print("\nWrote phase2_stage1_results.json + .md")
    return results


if __name__ == "__main__":
    main()
