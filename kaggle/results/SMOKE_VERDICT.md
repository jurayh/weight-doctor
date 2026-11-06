# Phase 2 smoke test — verdict (Nov 4, 2026)

**Verdict: PASS on the pre-registered smoke bar. Compute path settled
via Kaggle. Phase 2 itself has NOT started.**

Run: Kaggle free GPU, Qwen/Qwen2.5-0.5B-Instruct, torch 2.11.0+cu130,
Tesla T4, cuda_available true. Raw results in
`phase2_smoke_results.json` / `.md` in this folder.

## Against the pre-registered bar (kaggle/README.md)

- GPU present and named: PASS — Tesla T4.
- All 5 baseline prompts complete with token counts: PASS —
  29, 200, 126, 73, 4 tokens; 13.91 s total generation.
- Diff check completes, generation still runs: PASS — zeroed
  row 0 of `layers[0].self_attn.o_proj.weight` (1 row = 896
  entries, L2 0.68); rerun of the counting prompt is identical
  (29 tokens, 0.92 s). This proves edit-as-diff runs in torch
  on this GPU. It is not a fix, and the identical output is
  expected for such a tiny ablation on this prompt.
- Both result files returned: PASS.

## What the prompts actually showed (do not over-read)

- Loops detected: 1/5 — the banana prompt only. It asked for
  50 bananas; the model produced exactly 200 bananas and hit
  the 200-token cap without stopping. That is a stop/count
  failure on an instructed repetition, and the detector
  correctly flagged the repetition. It is NOT a clean
  unintended loop bug: the prompt told it to repeat. Phase 2
  needs a documented loop case where the model repeats
  without being instructed to, and a detector that
  distinguishes the two. The banana case must not be cited
  as a loop bug.
- The other prompts did not loop, but did not fully follow
  instructions either: counting stopped after 1–3 instead of
  counting to 100, the alphabet prompt stopped after A and B,
  the months prompt listed twice and stopped cleanly, the fox
  prompt completed the sentence in 4 tokens. Those are
  instruction-following gaps, not loop bugs, and are not
  Phase 2 cases on their own.

## Gate status after this run

- Phase 1 kills 1, 2, 4: PASS (Nov 3).
- Kill 3 (LLM doctor beats scripted baseline): still UNTESTED.
- Compute path: SETTLED for Kaggle (T4, $0, ~14 s for 5
  prompts). RunPod A5000 / $5 cap and Vast.ai spot remain
  fallbacks; VM CPU test remains paused.
- Phase 2 proper: NOT STARTED. Next step is to pin a
  documented, unintended loop case on this exact model,
  with target + breakage batteries, before any edit is
  attempted. No loop was claimed fixed by this smoke test.
