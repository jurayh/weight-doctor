# Weight Doctor Phase 2 pivot — Stage 2: candidate edits

**Status: build, Stage 2. No candidate has been evaluated
yet; this notebook is the evaluation.**

## What is in here

- `phase2_stage2.ipynb` — the one file to upload to
  Kaggle. Self-contained: the whole Stage 2 script is
  in one code cell.
- `phase2_stage2.py` — the same script, for reading /
  diffing. You do not need to upload this separately.
- `results/` — Stage 1 results and verdict, which this
  notebook stands on (see `results/STAGE1_VERDICT.md`).

## What Stage 2 does, in order

Pinned: Qwen/Qwen3.5-0.8B, float16, thinking ON, temp
0.6, top_p 0.95, top_k 20, seed 0, 1024-token cap,
presence_penalty attempted with the per-call fallback.

0. **Token-id resolution.** Runs the Sara prompt
   (closed in every probe) and walks its generated
   token ids until the cumulative text contains
   `</think>`. That token's id is the resolved id.
   Stage 1's trajectory id (248069) is NOT trusted;
   if no token is found here, the lm_head candidate
   is skipped, not failed.
1. **Baseline, this session.** Target 9 + breakage 14,
   scored close + correct. Expected 0/9 and 4/4
   known breakage; if it does not reproduce, the
   discrepancy is reported and candidates are scored
   against this session's baseline, not Stage 1's.
2. **Column scan.** Layer-15 MLP intermediate
   dimension split into 8 groups, each zeroed in turn
   (pre-hook on down_proj, weights unchanged) on
   Maria + France. Selection rule, frozen in the
   script: both-closed first, then fewest markers,
   then fewest tokens.
3. **Refinement.** The best group split into 4
   subgroups, same 2 prompts, same selection rule.
4. **Candidates, full battery each, receipts each:**
   - **C1** — best subgroup ablation, as a real
     weight edit (down_proj columns zeroed, run,
     restored): 112 columns ≈ 114,688 weights,
     0.015% — passes P2 on size.
   - **C2** — best group ablation, same method:
     448 columns ≈ 458,752 weights, 0.061% — passes
     P2 on size. Only if C1 does not pass both kills.
   - **C3–C5** — direction subtraction at layer 15,
     scales 0.25 / 0.5 / 1.0. Direction = mean L15
     MLP output at self-check tokens ("Wait"/"Okay")
     minus mean at other tokens, from a Maria run.
     This is a HOOK intervention, not a weight edit:
     its receipt says so, and a P1 pass here would
     still need a weight-edit conversion before
     claiming a surgical weight fix. Only if C1–C2 fail.
   - **C6** — lm_head `</think>` bias as a real
     weight edit (resolved row + alpha × mean-hidden
     unit vector, targeting +5 logit), only if all
     previous fail and Step 0 resolved an id.

**Stop rule:** evaluation stops at the first
candidate passing BOTH P1 (≥5/9 target, zero
breakage regressions vs this session's baseline)
and P2 (≤0.1% of weights, layer-limited). Later
families are recorded as "not evaluated", never as
failures. At most 6 candidate evaluations are in
this notebook (C1, C2, three direction scales, C6),
inside the plan's budget of 8.

## Run steps (your Kaggle account, same as before)

1. Create -> New Notebook -> File -> Import Notebook ->
   upload `phase2_stage2.ipynb`.
2. Settings: Accelerator -> GPU T4 x2, Internet -> On.
3. Run All. First run downloads the model, then:
   token resolution + baseline ~20 min, column scan
   ~10–15 min (16 generations, some close early),
   refinement ~5–8 min, then ~20 min per candidate
   battery. If C1 passes, the run is about 1 hour
   plus download. If it goes through all candidates,
   roughly 2.5–3 hours — still inside a 12-hour
   Kaggle session, and results are saved after every
   step, so a session end loses at most the step in
   progress.
4. Save Version -> Save & Run All, so the outputs survive.
5. **Before the session ends, save the files.**
   `/kaggle/working/` is per-session only: if the
   session ends without a save, the results and
   receipts are gone and cannot be recovered (this
   happened on the first Stage 2 run, Nov 4 — the
   console output survived, the files did not).
   Either download them from the notebook's file
   browser while the session is still alive, or
   File -> Save Version -> Quick Save (not Save &
   Run All) and download from the version's Output
   tab. Then send `phase2_stage2_results.json`,
   `phase2_stage2_results.md`, and the
   `phase2_stage2_receipts/` folder back here.

## Fix history

- Nov 4, 2026: the first Stage 2 run's direction
  derivation failed with "1024 captures vs 1024
  generated tokens". Cause: hook captures align
  1:1 with generated tokens (the prefill forward
  produces the first token); the code assumed a
  prompt capture plus one per token and skipped
  C3–C5. Fix: align on `zip(captures, gen_ids)`,
  and if lengths ever differ, align on the first
  min-length pairs with a warning instead of
  failing. The lm_head mean now uses all captures
  for the same reason (the first run used all but
  the first; the mean change is negligible, and
  the rerun's alpha will be comparable, not
  identical, to the first run's).

## Do not change the environment for this run

Do NOT pip-upgrade transformers before this run,
for the same comparability reason as Stage 1. The
one exception is the known accelerate ImportError
fix (new cell with
`!pip install -q -U accelerate transformers`, then
Session -> Restart, then Run All) — if needed, the
version change is logged in the results and must be
noted when the run is judged.

## What this notebook does NOT do

- It does not run the LLM doctor (Kill P3, still
  untested, needs a separate estimate-first paid-API
  approval).
- It does not claim anything beyond this model,
  mode, sampling, and battery (Kill P4).
- It does not treat a hook intervention as a weight
  edit, a closed-under-ablation prompt as
  close+correct, or a not-evaluated family as a
  failure.
