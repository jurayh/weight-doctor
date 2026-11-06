# Weight Doctor Phase 2 pivot — Stage 1: baseline + diagnosis

**Status: build, Stage 1 only. No weight edit is made here.
Candidate edits are Stage 2, built from this diagnosis.**

## What is in here

- `phase2_stage1.ipynb` — the one file to upload to
  Kaggle. It is self-contained: the whole Stage 1 script
  is in one code cell.
- `phase2_stage1.py` — the same script, for reading /
  diffing. You do not need to upload this separately.
- The torch-free harness lives in the repo at
  `weight_doctor/phase2/` (batteries, verifier, receipt).
  The Kaggle script copies its frozen prompts and success
  rules; the build validation checks the two against each
  other, so a drift between them would be caught before
  any run is judged.

## What Stage 1 does (and does not do)

Pinned: Qwen/Qwen3.5-0.8B, float16, thinking ON, temp
0.6, top_p 0.95, top_k 20, seed 0, 1024-token cap,
presence_penalty attempted with the per-call fallback.

1. **Baseline re-run.** Target (9) + breakage (14)
   prompts, scored close + correct. Must reproduce
   0/9 target and 4/4 known breakage, or the build
   stops: candidate edits are never evaluated against
   a baseline that does not hold. The 10 new breakage
   prompts' baseline rate is measured here, not assumed.
2. **Diagnosis (a) — marker rates.** Self-check marker
   counts ("Wait", "Okay", "Final Answer", "check",
   "confirm") in the baseline outputs, per 256-token
   block, for one math and one control prompt.
3. **Diagnosis (b) — </think> trajectory.** For 3
   prompts (one unclosed math, one unclosed control,
   one closed math), the rank and logit of the
   </think> token every 32 steps, from output_scores.
   Question: does the model ever seriously consider
   closing?
4. **Diagnosis (c) — thinking-OFF.** The 9 target
   prompts with thinking OFF. Diagnostic only, not
   scored: if they terminate cleanly, the failure is
   thinking-specific.
5. **Diagnosis (d) — layer ablation scan.** Layers
   0, 6, 12, 15, 19, 23 × 2 prompts (one math, one
   control), MLP output zeroed by a forward hook
   during generation. Hooks are removed after each
   layer. Weights are never changed; this is a
   diagnostic ablation, not an edit.

Deviation from the plan, stated in the script: the
plan listed "activation correlation + targeted
ablation" for the layer scan. Stage 1 does ablation
only, because ablation measures causal effect directly
and correlation alone would not decide a candidate
layer. If ablation identifies no layer, correlation
can be added in Stage 2 — deferred with a reason,
not silently dropped.

It does NOT run candidate edits, score any edit, or
claim any fix. Stage 2's candidate list (layers,
scales, families) will be frozen from these diagnosis
results and built as a separate notebook.

## Run steps (your Kaggle account, same as before)

1. Create -> New Notebook -> File -> Import Notebook ->
   upload `phase2_stage1.ipynb`.
2. Settings: Accelerator -> GPU T4 x2, Internet -> On.
3. Run All. First run downloads the ~1.6 GB model, then:
   baseline ~15–20 min (23 prompts at 1024 tokens,
   ~20 tok/s on reference kernels), trajectory ~3 min,
   thinking-OFF a few minutes (prompts should close
   early if thinking is the problem), ablation ~10 min
   (12 generations at up to 1024 tokens). Total roughly
   35–45 minutes of generation, plus download, well
   inside a 12-hour Kaggle session. Results are written
   after every step, so a session end loses at most
   the step in progress.
4. Save Version -> Save & Run All, so the outputs survive.
5. Download `phase2_stage1_results.json` and
   `phase2_stage1_results.md` from `/kaggle/working/`
   and send them back here.

## Fix history

- Nov 4, 2026: the first Stage 1 notebook crashed in the
  trajectory step with `AttributeError: 'Tensor' object
  has no attribute 'sequences'`. Cause: transformers
  `generate()` returns a plain Tensor unless
  `return_dict_in_generate=True`, even with
  `output_scores=True`; the code assumed scores implied
  a GenerateOutput. Fix: pass
  `return_dict_in_generate=True` when scores are asked
  for, handle a plain-Tensor return defensively (scores
  unavailable, warning, no crash), and wrap the whole
  trajectory step so a trajectory failure is recorded
  and thinking-OFF + ablation still run. Baseline and
  marker results from the crashed run were already saved
  to `phase2_stage1_results.json` in the working
  directory, but a re-uploaded notebook re-runs them
  from scratch (baseline ~15–20 min) — that is the
  cleanest rerun, and it also re-confirms the baseline.

## Do not change the environment for this run

Do NOT pip-upgrade transformers before this run:
comparability with the probes matters more than the
newest version. The one exception is the known
accelerate ImportError fix (new cell with
`!pip install -q -U accelerate transformers`, then
Session -> Restart, then Run All) — if that fix is
needed, the version change is logged in the results
and must be noted when the baseline is judged.
