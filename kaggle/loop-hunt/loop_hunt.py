"""Weight Doctor Phase 2 — loop hunt (Kaggle free GPU).

This is a CASE-FINDING probe, not Phase 2 itself. It answers one
question before any Phase 2 edit is attempted:

  Is there a clean, reproducible, UNINTENDED loop case on a
  Qwen-class small model, with a prompt that does NOT ask the
  model to repeat anything?

Why two tracks, and why the model switch is on the table:

  * The smoke test (Nov 4) pinned Qwen/Qwen2.5-0.5B-Instruct.
    Its only loop flag was the banana prompt, which INSTRUCTED
    repetition ("repeat 50 times") and is therefore NOT a clean
    loop bug. No documented unintended-loop case was found for
    Qwen2.5-0.5B-Instruct in the Nov 4 search.
  * The documented loop case is a different model:
    Qwen/Qwen3.5-0.8B in thinking mode. Its model card warns it
    is prone to repetition loops, and the "Sparks of Weight-Space
    Programming" post (MIT CSAIL, Nov 2026, blog at
    rikiyat.github.io/blog/weight-space-programming/) reports
    85 of its first 96 practice math answers running to the
    32,768-token limit, repeating and never reaching an answer,
    with model-card sampling (temperature 0.6, top_p 0.95,
    top_k 20, presence_penalty 1.5). A third-party rebuilt
    edited checkpoint exists, but this probe NEVER loads it:
    the probe is base-model only, so any later Phase 2 edit
    cannot be accused of copying that edit. The edited
    checkpoint is corroboration that the base loops, nothing
    more, and it is not an input to this probe.

Track A — empirical hunt, pinned smoke model:
  Qwen/Qwen2.5-0.5B-Instruct, greedy decoding, max 512 tokens.
  10 open-ended prompts, none of which asks to repeat, count,
  or list anything again. Plus 3 short-QA controls.

Track B — documented case, candidate switch:
  Qwen/Qwen3.5-0.8B, thinking mode ON, model-card sampling
  (temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5
  if the installed transformers accepts it, otherwise recorded
  as omitted), max 1024 tokens. 10 fresh GSM-style word
  problems (written for this probe, not copied from a test
  set), each with one numeric answer, none asking for
  repetition. Plus 3 short-QA controls, same sampling.
  Cap is 1024, not Astra's 32768: a true loop attractor
  should already be repeating well before 1024 tokens. If a
  prompt only starts looping after 1024, this probe will miss
  it, and that limit is recorded here, not hidden.
  IMPORTANT: Track B does NOT use greedy decoding. Qwen's
  own docs warn greedy can cause endless repetitions in
  thinking models, so a greedy loop on Track B would be a
  decoding artifact, not a model bug. Sampling is the point.

Detector v2 (stricter than the smoke detector, and the same
for both tracks): a generation counts as LOOP only if
  * it hits the token cap without emitting EOS, AND
  * a full line repeats 4+ times, OR
  * a 20-word window repeats 3+ times, OR
  * a 10-word window repeats 5+ times.
A short repeated phrase that stops cleanly is NOT a loop.
The prompt guard below is the other half: every probe prompt
is checked to contain no repeat/count-again instruction, so
a flag cannot be a banana-style false positive. If a prompt
fails the guard, the script stops instead of silently
running it.

Clean-case criteria (pre-registered, computed by this script
and reported, not judged by hand after the fact):
  * Track B clean case: >= 4/10 math prompts LOOP-flagged,
    0/3 controls LOOP-flagged, AND at least one math prompt
    LOOP-flagged again on a second run with the seed reset.
  * Track A clean case: >= 1/10 hunt prompts LOOP-flagged,
    0/3 controls LOOP-flagged, AND that prompt LOOP-flagged
    again on a second (greedy, hence deterministic) run.
  * Decision rule, fixed in advance: if Track B passes, Phase 2
    switches to Qwen3.5-0.8B with the math loop rate as the
    target battery. Else if Track A passes, Phase 2 stays on
    Qwen2.5-0.5B-Instruct with that prompt as the target case.
    Else neither passes: Phase 2 does NOT start on a loop
    case, and the negative result is the result. Do not force
    a case, do not lower the bar after seeing the outputs.

This probe makes NO weight edit. Track B also dumps the
model structure (layer count, layer path, layers 15 and 19
MLP down_proj shapes if present) so a later doctor can be
designed against the real architecture. Dumping is not
editing, and it does not copy Astra's edit (which layers and
columns to change are a Phase 2 doctor's job to find, with
its own target/breakage batteries).

Outputs: loop_hunt_results.json and loop_hunt_results.md
in the working directory (on Kaggle: /kaggle/working/).
Save the notebook version (Save & Run All) so the outputs
survive the session.

Troubleshooting: same as the smoke test. If you get the
accelerate ImportError, add a cell at the top with
  !pip install -q -U accelerate transformers
then Session -> Restart, then Run All again. The partially
initialized module stays broken until restart.
"""

import json
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- prompts

TRACK_A_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRACK_A_MAX_TOKENS = 512

TRACK_A_HUNT = [
    "Write a short story about a lighthouse keeper who finds a key in the sand.",
    "Explain how a bicycle works, in simple terms.",
    "Write a paragraph about why the seasons change.",
    "Describe a small town market on a Saturday morning.",
    "Write a letter to a friend about moving to a new city.",
    "Explain the difference between a lake and a river.",
    "Write an essay about the history of bread, starting with ancient grains.",
    "Write a product description for a wooden chair.",
    "Explain why the sky looks blue during the day.",
    "Write a short biography of a fictional inventor who built a machine that sorts socks.",
]

TRACK_A_CONTROLS = [
    "What is the capital of France? Answer in one sentence.",
    "What is 2 + 3? Answer with just the number.",
    "What colour is the sky on a clear day? Answer in one sentence.",
]

TRACK_B_MODEL = "Qwen/Qwen3.5-0.8B"
TRACK_B_MAX_TOKENS = 1024
# Model-card / Astra sampling for thinking mode. presence_penalty
# is attempted and recorded; if transformers rejects it, the run
# continues without it and says so (see generate_track_b).
TRACK_B_SAMPLING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
}

# Fresh GSM-style problems, written for this probe. Answers are
# recorded here so a later battery can score them; the probe
# itself does not score answers, it only detects loops.
TRACK_B_MATH = [
    ("Maria has 24 apples. She gives 7 to her brother and buys 15 more. How many apples does she have now?", 32),
    ("A baker makes 5 trays with 12 muffins each. He sells 38 muffins. How many muffins are left?", 22),
    ("Tom reads 18 pages a day. How many pages does he read in 2 weeks?", 252),
    ("A rectangle is 9 cm long and 4 cm wide. What is its area in square cm?", 36),
    ("Sara has 3 boxes with 14 pencils each, plus 5 loose pencils. How many pencils does she have in total?", 47),
    ("A train travels 60 km per hour for 3 hours, then travels 40 km more. How far did it travel in total, in km?", 220),
    ("Jake earns $8 per hour and works 6 hours a day, 5 days a week. How much does he earn in a week, in dollars?", 240),
    ("There are 48 students. Each table seats 6 students. How many tables are needed?", 8),
    ("A store sells notebooks for $3 each. Ana buys 7 notebooks and pays with $25. How much change does she get, in dollars?", 4),
    ("A garden has 5 rows with 11 plants each. 9 plants die. How many plants are left?", 46),
]

TRACK_B_CONTROLS = TRACK_A_CONTROLS

# Guard: no prompt may instruct repetition. Banana-style prompts
# are excluded by construction, and this check enforces it.
BANNED_PHRASES = [
    "repeat", "again, then again", "list them again",
    "count from", "one number per line", "50 times",
    "write it again", "then write it again",
]


def check_guard(prompts):
    bad = [p for p in prompts if any(b in p.lower() for b in BANNED_PHRASES)]
    if bad:
        raise ValueError(f"prompt guard failed, prompts instruct repetition: {bad}")


# ---------------------------------------------------------------- detector

def detect_loop(text):
    """Return (is_loop_pattern, rule, unit). Pattern only; the
    caller also requires hitting the token cap."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        unit, n = Counter(lines).most_common(1)[0]
        if n >= 4:
            return True, "line repeated 4+ times", unit[:200]
    words = text.split()
    for window, thresh in ((20, 3), (10, 5)):
        if len(words) >= window * thresh:
            # +1: the last window starts at len-window, inclusive.
            # The smoke detector used range(len-window) and so
            # needed one extra repeat to fire; v2 fixes that.
            windows = [" ".join(words[i:i + window])
                       for i in range(len(words) - window + 1)]
            unit, n = Counter(windows).most_common(1)[0]
            if n >= thresh:
                return True, f"{window}-word window repeated {thresh}+ times", unit[:200]
    return False, "no repeated line or window at the thresholds", None


# ---------------------------------------------------------------- generation

def build_text(tok, prompt, enable_thinking=None):
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tok.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Older template / tokenizer without enable_thinking.
        kwargs.pop("enable_thinking", None)
        return tok.apply_chat_template(messages, **kwargs)


def generate(model, tok, prompt, device, max_tokens, gen_kwargs,
             enable_thinking=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    text = build_text(tok, prompt, enable_thinking=enable_thinking)
    inputs = tok([text], return_tensors="pt").to(device)
    t0 = time.time()
    # transformers on Kaggle rejects presence_penalty with
    # ValueError ("model_kwargs are not used"), not TypeError.
    # The first version of this probe only caught TypeError, so
    # that error killed the whole notebook (seen Nov 4, 2026).
    # Fall back inside generate(), so one bad kwarg cannot lose
    # the track already run: retry without presence_penalty and
    # record the omission in the result.
    gen = dict(gen_kwargs)
    presence_penalty_used = "presence_penalty" in gen
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id, **gen)
    except (TypeError, ValueError) as e:
        if "presence_penalty" not in gen:
            raise
        print(f"presence_penalty rejected by this transformers "
              f"version ({e}); retrying without it")
        del gen["presence_penalty"]
        presence_penalty_used = False
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id, **gen)
    elapsed = time.time() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    hit_cap = len(new_tokens) >= max_tokens
    pattern, rule, unit = detect_loop(decoded)
    loop = bool(pattern and hit_cap)
    return {
        "prompt": prompt,
        "output": decoded,
        "output_tail": decoded[-300:],
        "new_tokens": int(len(new_tokens)),
        "hit_token_cap": bool(hit_cap),
        "seconds": round(elapsed, 2),
        "loop_detected": loop,
        "loop_rule": rule if loop else "not a loop (see rule)",
        "loop_unit": unit if loop else None,
        "presence_penalty_used": presence_penalty_used,
    }


def model_structure(model):
    """Dump-only architecture info for Track B. No editing."""
    info = {"total_params": sum(p.numel() for p in model.parameters())}
    # Qwen3.5-0.8B may sit under .model.layers or
    # .language_model.layers (VL wrapper). Record what is real.
    layers = None
    path = None
    for cand in ("model.layers", "language_model.layers",
                 "model.language_model.layers"):
        obj = model
        try:
            for part in cand.split("."):
                obj = getattr(obj, part)
            layers = obj
            path = cand
            break
        except AttributeError:
            continue
    info["layers_path"] = path
    if layers is not None:
        info["num_layers"] = len(layers)
        for idx in (15, 19):
            if idx < len(layers):
                mlp = getattr(layers[idx], "mlp", None)
                down = getattr(mlp, "down_proj", None) if mlp else None
                info[f"layer_{idx}_mlp_down_proj_shape"] = (
                    list(down.weight.shape) if down is not None else None)
    return info


# ---------------------------------------------------------------- main

def run_track(model_id, prompts, controls, max_tokens, gen_kwargs,
              device, enable_thinking, label):
    print(f"\n===== {label}: {model_id} =====")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device)
    model.eval()
    used_sampling = dict(gen_kwargs)
    if label.startswith("Track B"):
        # presence_penalty may be rejected by some transformers
        # versions; try it, and record a clean fallback.
        try:
            _ = model.generation_config
        except Exception:
            pass
    results = {"model_id": model_id, "hunt": [], "controls": []}
    all_prompts = prompts + controls
    check_guard(all_prompts)
    for i, prompt in enumerate(prompts):
        r = generate(model, tok, prompt, device, max_tokens, gen_kwargs,
                     enable_thinking=enable_thinking,
                     seed=0 if enable_thinking is not None else None)
        # Track B sampling is seeded, so a second run with the
        # seed reset is a reproducibility check. Track A greedy
        # is deterministic by construction; re-run it too, so
        # both tracks report the same reproducibility field.
        r2 = generate(model, tok, prompt, device, max_tokens, gen_kwargs,
                      enable_thinking=enable_thinking,
                      seed=0 if enable_thinking is not None else None)
        r["rerun_loop_detected"] = r2["loop_detected"]
        r["rerun_new_tokens"] = r2["new_tokens"]
        results["hunt"].append(r)
        print(f"[{label} {i}] tokens={r['new_tokens']} cap={r['hit_token_cap']} "
              f"loop={r['loop_detected']} rerun_loop={r2['loop_detected']} "
              f"({r['loop_rule']}) {r['seconds']}s")
    for i, prompt in enumerate(controls):
        r = generate(model, tok, prompt, device, max_tokens, gen_kwargs,
                     enable_thinking=enable_thinking,
                     seed=0 if enable_thinking is not None else None)
        results["controls"].append(r)
        print(f"[{label} control {i}] tokens={r['new_tokens']} "
              f"cap={r['hit_token_cap']} loop={r['loop_detected']} "
              f"{r['seconds']}s")
    if label.startswith("Track B"):
        results["structure"] = model_structure(model)
        print(f"Structure: {results['structure']}")
    results["used_sampling"] = used_sampling
    del model, tok
    torch.cuda.empty_cache()
    return results


def verdict(track, hunt, controls, min_hunt):
    hunt_loops = sum(1 for r in hunt if r["loop_detected"])
    control_loops = sum(1 for r in controls if r["loop_detected"])
    reproduced = [r for r in hunt
                  if r["loop_detected"] and r.get("rerun_loop_detected")]
    passed = (hunt_loops >= min_hunt and control_loops == 0
              and len(reproduced) >= 1)
    return {
        "hunt_loops": hunt_loops,
        "hunt_total": len(hunt),
        "control_loops": control_loops,
        "control_total": len(controls),
        "reproduced_loops": len(reproduced),
        "clean_case": passed,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }
    print(json.dumps(info, indent=2))

    # Track A: greedy, no thinking flag (2.5-Instruct has none).
    a = run_track(TRACK_A_MODEL, TRACK_A_HUNT, TRACK_A_CONTROLS,
                  TRACK_A_MAX_TOKENS, {"do_sample": False}, device,
                  enable_thinking=None, label="Track A")
    a_verdict = verdict("A", a["hunt"], a["controls"], min_hunt=1)
    # Save Track A immediately, so a later Track B failure can
    # never lose it (the original probe only wrote at the end).
    with open("loop_hunt_track_a.json", "w") as f:
        json.dump({"environment": info, "track_a": a,
                   "track_a_verdict": a_verdict}, f, indent=2)

    # Track B: model-card sampling, thinking on. generate()
    # already falls back inside each call if presence_penalty
    # is rejected (ValueError on Kaggle, seen Nov 4, 2026).
    # This outer catch is a second net: if Track B still fails
    # technically, that is a probe failure, NOT a clean-case
    # failure, and it must be recorded as such.
    b_sampling = dict(TRACK_B_SAMPLING)
    b_error = None
    try:
        b = run_track(TRACK_B_MODEL,
                      [p for p, _ in TRACK_B_MATH], TRACK_B_CONTROLS,
                      TRACK_B_MAX_TOKENS, b_sampling, device,
                      enable_thinking=True, label="Track B")
    except (TypeError, ValueError) as e:
        print(f"Track B failed technically, retrying without "
              f"presence_penalty: {e}")
        b_sampling.pop("presence_penalty", None)
        try:
            b = run_track(TRACK_B_MODEL,
                          [p for p, _ in TRACK_B_MATH], TRACK_B_CONTROLS,
                          TRACK_B_MAX_TOKENS, b_sampling, device,
                          enable_thinking=True, label="Track B")
        except Exception as e2:
            b_error = str(e2)
            b = {"error": b_error, "hunt": [], "controls": []}
    if b_error is None:
        b["used_sampling"] = b_sampling
        # Actual sampling is what the first hunt result used,
        # after any per-call fallback.
        if b["hunt"]:
            used = dict(b_sampling)
            if not b["hunt"][0].get("presence_penalty_used", True):
                used.pop("presence_penalty", None)
            b["used_sampling"] = used
        b["math_answers"] = [ans for _, ans in TRACK_B_MATH]
        b_verdict = verdict("B", b["hunt"], b["controls"], min_hunt=4)
    else:
        b_verdict = {"hunt_loops": 0, "hunt_total": 0, "control_loops": 0,
                     "control_total": 0, "reproduced_loops": 0,
                     "clean_case": False}

    if b_error is not None:
        decision = ("Probe incomplete: Track B failed technically "
                    f"({b_error}). This is not a clean-case failure; "
                    "Track A alone does not settle Phase 2.")
    elif b_verdict["clean_case"]:
        decision = ("Track B clean case: Phase 2 switches to "
                    "Qwen3.5-0.8B with the math loop rate as target battery.")
    elif a_verdict["clean_case"]:
        decision = ("Track A clean case: Phase 2 stays on "
                    "Qwen2.5-0.5B-Instruct with the flagged prompt as target case.")
    else:
        decision = ("No clean case: Phase 2 does NOT start on a loop "
                    "case. The negative result stands; do not lower the bar.")
    print(f"\nTrack A verdict: {a_verdict}")
    print(f"Track B verdict: {b_verdict}")
    print(f"Decision: {decision}")

    results = {"environment": info, "track_a": a, "track_a_verdict": a_verdict,
               "track_b": b, "track_b_verdict": b_verdict,
               "decision": decision}
    with open("loop_hunt_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("loop_hunt_results.md", "w") as f:
        f.write("# Weight Doctor Phase 2 — loop hunt\n\n")
        f.write(f"GPU: {info['gpu_name'] or info['device']}\n\n")
        f.write(f"Track A (Qwen2.5-0.5B-Instruct, greedy): "
                f"{a_verdict['hunt_loops']}/{a_verdict['hunt_total']} loops, "
                f"controls {a_verdict['control_loops']}/{a_verdict['control_total']}, "
                f"clean case: {a_verdict['clean_case']}\n\n")
        f.write(f"Track B (Qwen3.5-0.8B, thinking, model-card sampling): "
                f"{b_verdict['hunt_loops']}/{b_verdict['hunt_total']} loops, "
                f"controls {b_verdict['control_loops']}/{b_verdict['control_total']}, "
                f"clean case: {b_verdict['clean_case']}\n\n")
        f.write(f"Decision: {decision}\n\n")
        f.write("This is a case-finding probe, not Phase 2. No loop was "
                "claimed fixed, and no weight was edited.\n")
    print("\nWrote loop_hunt_results.json + .md")
    return results


if __name__ == "__main__":
    main()
