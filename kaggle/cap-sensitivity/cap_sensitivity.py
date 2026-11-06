"""Weight Doctor — cap-sensitivity probe (Kaggle free GPU).

This is a GATE probe for a possible Phase 2 pivot, not Phase 2
itself. It answers one question, and only one:

  Was Track B's "thinking never terminates" a real failure,
  or just an artifact of the 1024-token cap?

Background (loop hunt, Nov 4, 2026): Qwen/Qwen3.5-0.8B in
thinking mode, at a 1024-token cap, closed thinking on only
4/13 prompts (3 math + 1 control). The other 9 — 7 math
problems and 2 trivial controls — hit the cap still inside
thinking, even though the right answer was already in the
thinking text. That could be a genuine non-termination
failure, or it could be that this model simply thinks
verbosely and would have closed at 1500 or 3000 tokens.
Building an edit for a cap artifact would be building for
nothing, so the cap is the one variable this probe changes.

What this probe does:
  * Same model, same thinking mode, same sampling, same
    seed (0) as loop-hunt Track B, so the first 1024 tokens
    should reproduce Track B's run. Sampling is temperature
    0.6, top_p 0.95, top_k 20; presence_penalty is attempted
    and, as on the Nov 4 Kaggle image, is expected to be
    rejected with ValueError — the per-call fallback retries
    without it and records the omission, exactly as in the
    loop hunt. Do NOT pip-upgrade transformers for this
    probe: a version change would confound the cap effect
    with an environment effect, which is the thing this
    probe is trying to isolate.
  * Cap 4096 tokens (not 32768: if it has not closed by
    4096 on a trivial arithmetic problem, calling it
    "long thinking" is no longer credible, and 4096 keeps
    the run inside a Kaggle session at reference-kernel
    speed — about 20 tokens/s on a T4 in the loop hunt, so
    worst case ~35-40 minutes of generation for this probe).
  * The 9 non-terminating prompts from Track B, verbatim
    (checked against loop_hunt_results.json; the script
    stops if a prompt does not match the Track B list).
    Plus the 4 already-closed prompts as sanity controls:
    they should close at the same token counts as before
    (666, 350, 732, 134), proving the longer cap did not
    change behaviour on prompts that already terminated.
  * For each prompt: closed = "</think>" appears in the
    output AND generation stopped before the cap. If it
    closes, a second seeded run confirms closure (cheap,
    because it closes early). Prompts that hit the 4096 cap
    are NOT re-run: a second 4096-token run adds nothing
    to the gate and costs the most time.
  * Results are written after EVERY prompt
    (cap_sensitivity_results.json), so a crash or a
    session end cannot lose completed prompts — the
    loop-hunt lesson, applied here from the start.

Pre-registered gate (fixed before the run, not judged
after seeing the outputs):
  * PIVOT: >= 5 of the 9 non-terminating prompts are still
    unclosed at 4096 tokens. Then non-termination is real,
    and Phase 2 may pivot to it, with close-rate at 1024
    as the target battery and answer accuracy + short-QA
    as breakage.
  * NO PIVOT: <= 4 of the 9 are still unclosed at 4096
    (i.e. at least 5 closed). Then the 1024-cap reading was
    mostly a cap artifact, the pivot dies, and the loop
    line stays closed. A 3-4 unclosed result is still
    NO PIVOT under this rule — it is not upgraded to a
    pivot by re-reading it afterwards.
  * Sanity failure: if an already-closed prompt fails to
    close at 4096, the probe is INCOMPLETE (environment
    changed), not a gate result, and must be re-run before
    anything is judged.

This probe makes NO weight edit. It does not score whether
the final answer is right beyond recording, for closed
prompts, whether the expected answer appears after
</think> — that is breakage-battery material for later,
not the gate.

Outputs: cap_sensitivity_results.json and
cap_sensitivity_results.md in the working directory
(on Kaggle: /kaggle/working/). Save the notebook version
(Save & Run All) so the outputs survive the session.

Troubleshooting: same as the loop hunt. If you get the
accelerate ImportError, add a cell at the top with
  !pip install -q -U accelerate transformers
then Session -> Restart, then Run All again. Note: that
fix changes the transformers version, so if it is needed,
record it in the results — the version is logged at the
start of every run for exactly this reason.
"""

import json
import time
from importlib import metadata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_TOKENS = 4096
SAMPLING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
}

# Verbatim Track B prompts (loop hunt, Nov 4). The first list
# is the 9 that hit the 1024 cap without closing thinking;
# the second list is the 4 that closed, as sanity controls.
# Expected answers are recorded for closed-prompt checking
# only; the gate itself is closure, not accuracy.
NON_TERMINATING = [
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

ALREADY_CLOSED = [
    ("Sara has 3 boxes with 14 pencils each, plus 5 loose pencils. How many pencils does she have in total?", "47", 666),
    ("There are 48 students. Each table seats 6 students. How many tables are needed?", "8", 350),
    ("A garden has 5 rows with 11 plants each. 9 plants die. How many plants are left?", "46", 732),
    ("What is 2 + 3? Answer with just the number.", "5", 134),
]

BANNED_PHRASES = [
    "repeat", "again, then again", "list them again",
    "count from", "one number per line", "50 times",
    "write it again", "then write it again",
]


def check_guard(prompts):
    bad = [p for p in prompts if any(b in p.lower() for b in BANNED_PHRASES)]
    if bad:
        raise ValueError(f"prompt guard failed: {bad}")


def build_text(tok, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate(model, tok, prompt, device, seed=0):
    torch.manual_seed(seed)
    text = build_text(tok, prompt)
    inputs = tok([text], return_tensors="pt").to(device)
    gen = dict(SAMPLING)
    presence_penalty_used = "presence_penalty" in gen
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS,
                pad_token_id=tok.eos_token_id, **gen)
    except (TypeError, ValueError) as e:
        if "presence_penalty" not in gen:
            raise
        print(f"presence_penalty rejected ({e}); retrying without it")
        del gen["presence_penalty"]
        presence_penalty_used = False
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS,
                pad_token_id=tok.eos_token_id, **gen)
    elapsed = time.time() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    hit_cap = len(new_tokens) >= MAX_TOKENS
    closed = ("</think>" in decoded) and not hit_cap
    return {
        "prompt": prompt,
        "output_tail": decoded[-300:],
        "new_tokens": int(len(new_tokens)),
        "hit_token_cap": bool(hit_cap),
        "closed": bool(closed),
        "seconds": round(elapsed, 2),
        "presence_penalty_used": presence_penalty_used,
    }


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
    check_guard([p for p, _ in NON_TERMINATING]
                + [p for p, _, _ in ALREADY_CLOSED])

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()

    results = {"environment": info, "non_terminating": [],
               "sanity_closed": []}

    def save():
        with open("cap_sensitivity_results.json", "w") as f:
            json.dump(results, f, indent=2)

    for prompt, expected in NON_TERMINATING:
        r = generate(model, tok, prompt, device, seed=0)
        r["expected"] = expected
        if r["closed"]:
            # Confirm closure on a second seeded run (cheap:
            # it closes early). Cap-hitters are not re-run.
            r2 = generate(model, tok, prompt, device, seed=0)
            r["rerun_closed"] = r2["closed"]
            r["rerun_new_tokens"] = r2["new_tokens"]
        else:
            r["rerun_closed"] = None
            r["rerun_new_tokens"] = None
        results["non_terminating"].append(r)
        save()
        print(f"[gate] closed={r['closed']} rerun_closed={r['rerun_closed']} "
              f"tokens={r['new_tokens']} {r['seconds']}s "
              f"prompt={prompt[:60]!r}")

    sanity_failed = []
    for prompt, expected, old_tokens in ALREADY_CLOSED:
        r = generate(model, tok, prompt, device, seed=0)
        r["expected"] = expected
        r["track_b_tokens"] = old_tokens
        results["sanity_closed"].append(r)
        save()
        if not r["closed"]:
            sanity_failed.append(prompt)
        print(f"[sanity] closed={r['closed']} tokens={r['new_tokens']} "
              f"(Track B: {old_tokens}) {r['seconds']}s "
              f"prompt={prompt[:60]!r}")

    unclosed = sum(1 for r in results["non_terminating"] if not r["closed"])
    if sanity_failed:
        decision = ("Probe INCOMPLETE: an already-closed prompt failed "
                    f"to close at 4096 ({len(sanity_failed)} failed). "
                    "Environment changed; re-run before judging the gate.")
        gate = "incomplete"
    elif unclosed >= 5:
        decision = ("PIVOT: non-termination is real — "
                    f"{unclosed}/9 still unclosed at 4096 tokens. "
                    "Phase 2 may pivot to thinking non-termination, "
                    "with close-rate at 1024 as target battery.")
        gate = "pivot"
    else:
        decision = ("NO PIVOT: only "
                    f"{unclosed}/9 still unclosed at 4096 tokens. "
                    "The 1024-cap reading was mostly a cap artifact; "
                    "the pivot dies and the loop line stays closed.")
        gate = "no_pivot"
    results["gate"] = gate
    results["unclosed_at_4096"] = unclosed
    results["decision"] = decision
    save()
    print(f"\nUnclosed at 4096: {unclosed}/9, gate: {gate}")
    print(f"Decision: {decision}")

    with open("cap_sensitivity_results.md", "w") as f:
        f.write("# Weight Doctor — cap-sensitivity probe\n\n")
        f.write(f"Model: {MODEL_ID} on {info['gpu_name'] or device}, "
                f"cap {MAX_TOKENS} tokens\n\n")
        f.write(f"Non-terminating prompts still unclosed at 4096: "
                f"{unclosed}/9\n\n")
        f.write(f"Sanity (already-closed) failures: {len(sanity_failed)}\n\n")
        f.write(f"Gate: {gate}\n\n")
        f.write(f"Decision: {decision}\n\n")
        f.write("This is a gate probe, not Phase 2. No weight was edited.\n")
    print("\nWrote cap_sensitivity_results.json + .md")
    return results


if __name__ == "__main__":
    main()
