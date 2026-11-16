# Weight Doctor Phase 2 — stronger-forcing probe (C7)

**Status: build. Nothing has been evaluated yet;
this notebook is the test.**

## Question and decision rule (frozen before the run)

Does a stronger static bias close hard problems, and
at what cost on easy ones? C6 (target +5 logit)
closes 5/9 easy problems but 0/9 hard ones — it does
not close at all on hard problems. This probe tests
two stronger alphas, C7a (+10, double C6) and C7b
(+20, quadruple C6), same lm_head-row method.

For each alpha, let H = close+correct on HARD,
E = close+correct on TARGET, R = breakage
regressions against the in-run baseline.

- **EXTENDS** iff H ≥ 5/9 AND E ≥ 5/9 AND R = 0.
- **TRADEOFF** iff H ≥ 5/9 but E < 5/9 or R > 0.
- **LIMIT** iff H < 5/9 for that alpha. If both
  alphas hit LIMIT, static forcing hits its limit.

Each alpha is judged on its own; a split between
them is itself the result.

## Batteries

- **HARD** (9 multi-step problems, frozen Nov 7 in
  `C6_HARD_PROBE_PLAN.md`, same as the hard probe).
- **TARGET** (9 original target prompts).
- **BREAKAGE** (14 original breakage prompts).
- **Gate:** C6 (+5) re-run on TARGET first. It must
  reproduce 5/9 or the run STOPS — the setup
  changed and nothing else in the run would be
  comparable.

The baseline on all three batteries is measured
in-run, not assumed (Stage 2's was 0/9 targets,
8/14 breakage).

## What the probe does

Pinned: Qwen/Qwen3.5-0.8B, float16, thinking ON,
temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token
cap, presence_penalty attempted with the per-call
fallback. Alphas are derived in-run from Sara's
mean hidden (alpha = target / ||mean_hidden||),
same method as Stage 2; expected alphas are ~0.1731
(+10) and ~0.3463 (+20) from the hard probe's
measured projection.

0. Re-resolves the `</think>` token id from a closed
   Sara generation (same method as Stage 2). Stops
   if no id resolves.
1. Baseline on HARD + TARGET + BREAKAGE (32
   prompts), thinking ON. Raw outputs and token
   ids saved.
2. Gate: C6 (+5) on TARGET. Must be 5/9 or stop.
3. C7a (+10) on all three batteries, with its own
   receipt.
4. C7b (+20) on all three batteries, with its own
   receipt.
5. First-appearance analysis per alpha (read-only,
   from saved outputs): for the prompts each alpha
   closes, did closure come before or after
   derivation? Diagnostic only, not a kill bar.

A crashing candidate is recorded and skipped; it
cannot kill the run. Both alphas are evaluated
regardless of outcome — there is no stop rule
between them.

## Run steps (your Kaggle account, same as before)

1. Create -> New Notebook -> File -> Import Notebook ->
   upload `c7_probe.ipynb`.
2. Settings: Accelerator -> GPU T4 x2, Internet -> On.
3. Run All. Baseline ~20 min (32 prompts), gate ~5 min,
   each alpha ~20 min (32 prompts), plus derivation
   and analysis. Total roughly 60–90 minutes plus
   model download, well inside a 12-hour session.
   Results save after every step.
4. **Save the files before the session ends** —
   download `c7_probe_results.json`,
   `c7_probe_results.md`, and `c7_probe_receipts/`
   from the file browser while the session is alive,
   or File -> Save Version -> Quick Save (not Save &
   Run All) and download from the version's Output
   tab. Kaggle will not download a folder directly:
   zip the receipts folder first with
   `!cd /kaggle/working && zip -r c7_probe_results.zip c7_probe_results.json c7_probe_results.md c7_probe_receipts/`,
   then download the zip.
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
