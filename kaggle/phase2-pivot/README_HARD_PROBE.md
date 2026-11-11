# Weight Doctor Phase 2 — C6 hard-problem probe

**Status: build. Nothing has been evaluated on the
hard battery yet; this notebook is the test.**

## Question and decision rule (frozen before the run)

Does C6 generalise to problems where the thinking does
real work? C6 "generalises" on easy arithmetic, where
the thinking is wasted second-guessing. On a multi-step
problem derivation takes longer, and C6's lm_head bias
does not wait for the answer — it just raises the
`</think>` probability. The crux is timing: does
closure come before or after derivation?

Let H = C6 close+correct on the hard battery, T =
thinking-OFF accuracy on the same 9 hard problems.

- **Generalisation extends to hard problems** iff
  H ≥ 5/9 AND H ≥ T.
- **Forcing closure hurts** iff H < T. The
  generalisation claim then narrows to easy problems,
  and the README must say so.
- **Partial** otherwise (T ≤ H < 5/9).

## The hard battery

9 multi-step problems (fractions then subtractions,
average speed over legs, chained dependencies, a
cheapest-option comparison, dozen conversions), each
with a single-number answer, frozen Nov 7, 2026 in
`C6_HARD_PROBE_PLAN.md` (see `HARD` in the script).
No overlap with the original 23 or the held-out 17,
arithmetic checked by hand at writing time. The
baseline pass rate on this battery is measured in
the run, not assumed.

## What the probe does

Pinned: Qwen/Qwen3.5-0.8B, float16, thinking ON,
temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token
cap, presence_penalty attempted with the per-call
fallback. C6 is the same alpha method as Stage 2
(mean hidden from Sara, target +5 logit,
tied-embedding status recorded).

0. Re-resolves the `</think>` token id from a closed
   Sara generation (same method as Stage 2). Stops
   if no id resolves.
1. Baseline, thinking ON, on the hard battery (9
   prompts). Raw outputs and token ids saved.
2. Thinking-OFF diagnostic on the same 9 hard
   prompts (same method as Stage 1).
3. Derives C6 (same method as Stage 2).
4. **Control first:** C6 on the original 9 target
   prompts, expecting ~5/9. If it scores below 3/9
   the run STOPS — the setup changed and the hard
   numbers would not be comparable. (The plan's step
   list has hard before control; the control is
   evaluated first so the stop rule can actually
   stop the run. This reordering is the only
   deviation from the plan, and it preserves the
   rule's intent.)
5. C6 on the hard battery (9 prompts), with its own
   receipt.
6. First-appearance analysis (read-only, from saved
   outputs): in the unedited thinking traces, at
   what token does the expected answer first appear,
   if at all — and does C6's closure token come
   before or after it? Diagnostic only, not a kill
   bar.

Raw outputs are saved for every prompt this time
(the generalisation probe dropped them, so its
flags could not be re-checked from text).

## Run steps (your Kaggle account, same as before)

1. Create -> New Notebook -> File -> Import Notebook ->
   upload `hard_probe.ipynb`.
2. Settings: Accelerator -> GPU T4 x2, Internet -> On.
3. Run All. Baseline ~8 min (9 prompts), thinking-OFF
   ~1 min, C6 derivation ~1 min, control ~8 min,
   hard battery ~8 min, analysis ~1 min. Total
   roughly 30–45 minutes plus model download, well
   inside a 12-hour session. Results save after every
   step.
4. **Save the files before the session ends** —
   download `hard_probe_results.json`,
   `hard_probe_results.md`, and `hard_probe_receipts/`
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
