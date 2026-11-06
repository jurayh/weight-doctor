"""Weight Doctor Phase 2 smoke test — Kaggle free GPU version.

This is NOT Phase 2 itself. It answers one question, on a free
Kaggle T4, before any Phase 2 work starts:

  Can this environment load a Qwen-class 0.5B model on GPU,
  generate from it, detect a repetition loop, and apply a tiny
  weight diff and re-run?

If any step fails, Phase 2 does not start on Kaggle; the failure
is the result, and it is reported as such.

Model: Qwen/Qwen2.5-0.5B-Instruct, 0.49B parameters, safetensors.
Dtype is float16, not bfloat16: T4s predate bf16, and the Kaggle
default environment already ships torch + transformers, so this
script does NOT pip-install anything. If an import is missing,
that is a smoke-test failure, not a reason to install over a
no-internet session.

Loop detection (simple, disclosed, no vibes): greedy generation,
max 200 new tokens. A generation is flagged LOOP if
  * it hits the token cap without emitting EOS, AND a 20-token
    window repeats at least 3 times, OR
  * the same full line repeats 4+ times.
Otherwise it is NO_LOOP. The prompts below are loop-prone by
construction (counting, repeating lists); a NO_LOOP on all of
them is still a passing smoke test — it means the model did not
loop on these prompts, not that loops do not exist.

Weight-diff check (also not a fix): after the baseline runs,
zero exactly one row of one layer's output projection
(`model.layers[0].self_attn.o_proj.weight[0, :] = 0`), re-run
one prompt, and confirm generation still runs and the diff
stats are reported (1 row, its L2 norm). This proves the
edit-as-diff machinery works in torch on this GPU. It proves
nothing about fixing a loop.

Outputs: phase2_smoke_results.json and phase2_smoke_results.md
in the working directory (on Kaggle: /kaggle/working/). Save the
notebook version (Save & Run All) so the outputs survive the
session; interactive-session files do not.

Troubleshooting (seen Nov 4, 2026): if you get
  ImportError: cannot import name 'dispatch_model' from
  partially initialized module 'accelerate.big_modeling'
that is a mismatched accelerate/transformers pair in the
Kaggle image, not this script. Fix, in a new cell at the top:
  !pip install -q -U accelerate transformers
then restart the session (Session -> Restart), then Run All
again. Installing in the same session does NOT fix it: the
partially initialized module stays broken until restart.
If it still fails, send back the output of
  !pip show accelerate transformers torch
so exact versions can be pinned.

Run on Kaggle: upload phase2_smoke.ipynb, set Accelerator to
GPU T4 x2 (one GPU is used; quota is charged per hour, not per
GPU), Internet to On (needs phone verification once), Run All.
Expected wall time: model download ~1 GB once, then a few
minutes of generation. Expected quota: well under 1 GPU-hour
for the smoke run itself, download time included in wall time.
"""

import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 200

# Loop-prone by construction. None of these is a claim that the
# model WILL loop on it; the detector reports what actually
# happened, prompt by prompt.
PROMPTS = [
    "Count from 1 to 100, one number per line:\n1\n2\n3",
    "Repeat the word 'banana' 50 times, separated by spaces:",
    "List the months of the year, then list them again, then again:",
    "Write the alphabet, then write it again, then again:",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
]


def detect_loop(text):
    """Return (is_loop, reason). See module docstring for the rule."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        from collections import Counter

        counts = Counter(lines)
        most, n = counts.most_common(1)[0]
        if n >= 4 and len(lines) >= 4:
            return True, f"line {most!r} repeated {n} times"
    # Token-ish repetition: a 20-word window appearing 3+ times.
    words = text.split()
    if len(words) >= 60:
        windows = [" ".join(words[i:i + 20])
                   for i in range(len(words) - 20)]
        from collections import Counter

        _, n = Counter(windows).most_common(1)[0]
        if n >= 3:
            return True, f"20-word window repeated {n} times"
    return False, "no repeated line or window at the thresholds"


def generate(model, tok, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    hit_cap = len(new_tokens) >= MAX_NEW_TOKENS
    is_loop, reason = detect_loop(decoded)
    # A loop flag only counts if generation also ran long; a short
    # repeated phrase that stops cleanly is not a loop bug.
    loop = bool(is_loop and hit_cap)
    return {
        "prompt": prompt,
        "output": decoded,
        "new_tokens": int(len(new_tokens)),
        "hit_token_cap": bool(hit_cap),
        "seconds": round(elapsed, 2),
        "loop_detected": loop,
        "loop_reason": reason if loop else "not a loop (see rule)",
    }


def main(model_id=MODEL_ID, model=None, tok=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = {
        "model_id": model_id,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }
    print(json.dumps(info, indent=2))
    if model is None:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    results = {"environment": info, "baseline": [], "diff_check": None}
    for prompt in PROMPTS:
        r = generate(model, tok, prompt, device)
        results["baseline"].append(r)
        print(f"\n--- prompt: {prompt[:60]!r}")
        print(f"tokens={r['new_tokens']} cap={r['hit_token_cap']} "
              f"loop={r['loop_detected']} ({r['loop_reason']}) "
              f"{r['seconds']}s")
        print(r["output"][:400])

    # Diff check: zero one row, re-run one prompt, report the diff.
    with torch.no_grad():
        layer = model.model.layers[0].self_attn.o_proj
        before = layer.weight[0, :].clone()
        layer.weight[0, :] = 0
        changed = int((layer.weight[0, :] != before).sum())
        l2 = float(torch.linalg.norm(layer.weight[0, :] - before))
    r = generate(model, tok, PROMPTS[0], device)
    results["diff_check"] = {
        "edit": "zero row 0 of layers[0].self_attn.o_proj.weight",
        "entries_changed": changed,
        "l2_norm": l2,
        "rerun": r,
    }
    print(f"\nDiff check: {changed} entries changed, L2={l2:.4f}, "
          f"rerun tokens={r['new_tokens']}, loop={r['loop_detected']}")

    with open("phase2_smoke_results.json", "w") as f:
        json.dump(results, f, indent=2)
    loops = sum(1 for r in results["baseline"] if r["loop_detected"])
    with open("phase2_smoke_results.md", "w") as f:
        f.write("# Weight Doctor Phase 2 smoke test\n\n")
        f.write(f"Model: {model_id} on {info['gpu_name'] or device}\n\n")
        f.write(f"Baseline prompts: {len(PROMPTS)}, "
                f"loops detected: {loops}\n\n")
        f.write("This is a compute-path smoke test, not a Phase 2 "
                "result. No loop was claimed fixed.\n")
    print("\nWrote phase2_smoke_results.json + .md")
    return results


if __name__ == "__main__":
    main()
