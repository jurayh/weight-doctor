# Weight Doctor Phase 2 pivot — generalisation probe

**Status: build. No candidate has been evaluated on
the held-out battery yet; this notebook is the test.**

## Question and decision rule (frozen before the run)

Does the Stage 2 pass generalise, or was it a fit to
the 9+14 battery? C6 lm_head bias passed P1+P2
exactly at the bar (5/9, 0 regressions) on the frozen
battery; direction 0.5 was one regression away (5/9,
1 regression).

- **C6 generalises** if it scores ≥5/9 target with
  zero regressions on this held-out battery.
- Otherwise the Phase 2 pass **does not generalise**,
  and the verdict becomes "passes the frozen battery,
  does not generalise".
- Direction 0.5 is a comparator, judged by the same
  bar. Both candidates are evaluated regardless of
  outcome — there is no stop rule in this probe.

## The held-out battery

9 new target prompts + 8 new breakage prompts,
written and frozen Nov 5, 2026, before this run
(see `NEW_TARGET` / `NEW_BREAKAGE` in the script).
Same styles as the original batteries, all prompts
and answers new, no overlap with the original 23,
arithmetic checked by hand at writing time. The
baseline pass rate on this battery is measured in
the run, not assumed — if the baseline already
passes several targets, that is reported, not hidden.

## What the probe does

Pinned: Qwen/Qwen3.5-0.8B, float16, thinking ON,
temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token
cap, presence_penalty attempted with the per-call
fallback.

0. Re-resolves the `</think>` token id from a closed
   Sara generation (same method as Stage 2).
1. Baseline on the new battery (17 prompts).
2. Re-derives the layer-15 self-check direction
   from the original Maria prompt (the direction is
   part of the candidate definition, not battery
   leakage; the float16 dtype fix from Nov 5 is in).
3. Evaluates **C6 lm_head bias** (same alpha method
   as Stage 2: mean hidden from Sara, target +5
   logit, tied-embedding status recorded) on the
   new battery, with its own receipt.
4. Evaluates **direction 0.5** (hook, not a weight
   edit, disclosed in the receipt) on the new
   battery, with its own receipt.

A crashing candidate is recorded and skipped; it
cannot kill the run.

## Run steps (your Kaggle account, same as before)

1. Create -> New Notebook -> File -> Import Notebook ->
   upload `gen_probe.ipynb`.
2. Settings: Accelerator -> GPU T4 x2, Internet -> On.
3. Run All. Baseline ~15 min (17 prompts), direction
   derivation ~1 min, then ~15 min per candidate
   battery. Total roughly 45–60 minutes plus model
   download, well inside a 12-hour session. Results
   save after every step.
4. **Save the files before the session ends** —
   download `gen_probe_results.json`,
   `gen_probe_results.md`, and `gen_probe_receipts/`
   from the file browser while the session is alive,
   or File -> Save Version -> Quick Save (not Save &
   Run All) and download from the version's Output
   tab. Kaggle will not download a folder directly:
   zip the receipts folder first (a cell for that
   was used last time), then download the zip.
5. Send the files back here.

## Do not change the environment for this run

Do NOT pip-upgrade transformers before this run,
for the same comparability reason as Stage 2. The
one exception is the known accelerate ImportError
fix (new cell with
`!pip install -q -U accelerate transformers`, then
Session -> Restart, then Run All) — if needed, the
version change is logged in the results and must be
noted when the run is judged.
