# Stronger-forcing follow-up — plan (pre-registered)

Written Nov 13, 2026, before any run. The question, the
candidates, the batteries, and the decision rule are fixed
now; the probe build starts only on Yuriy's go, and the
run starts only after the probe is built.

## The question

C6 (target +5 logit, alpha 0.0866) closes 5/9 easy
problems but 0/9 hard ones — it does not close at all
on hard problems. The bias is too weak to overcome the
longer, more involved self-checking on multi-step
problems. The model derives the answer in thinking on
all 9 hard problems (first appearance at tokens
336-922) and never stops.

Does a stronger static bias close hard problems, and
at what cost on easy ones? A stronger bias should
close more — but it should also start closing easy
thinking too early, before derivation, and producing
wrong answers. The interesting result is the
tradeoff, not just the hard score. That is why the
decision rule below is two-sided.

## Candidates (same method as C6, new alphas)

Both are weight edits: one lm_head row, alpha =
target_logit_increase / ||mean_hidden||, mean hidden
from Sara, derived in-run exactly as Stage 2 did.
The tied-embedding disclosure applies to both, at
larger magnitude: the row edit also changes the
`</think>` input embedding.

- **C7a: target +10 logit.** Expected alpha ~0.1731
  (from the hard probe's measured proj, ||mean_hidden||
  = 5 / 0.08656 = 57.76; the probe re-derives alpha
  in-run, so this is an expectation, not a frozen
  number). Double C6.
- **C7b: target +20 logit.** Expected alpha ~0.3463.
  Quadruple C6.

Both candidates are evaluated regardless of outcome
(no stop rule between them), as the generalisation
probe did with C6 and direction 0.5.

Explicitly out of this probe: a growing bias (a logit
bonus that increases with thinking length). That is a
different mechanism — a decoding-time intervention,
not a weight edit — and it belongs in its own plan if
the static tradeoff here turns out bad.

## Batteries

- **HARD** (9 multi-step problems, same as the hard
  probe, frozen Nov 7 in C6_HARD_PROBE_PLAN.md).
- **TARGET** (9 original target prompts).
- **BREAKAGE** (14 original breakage prompts) — the
  no-regression check, which the hard probe did not
  include. Stage 2's measured baseline was 8/14
  passing; this probe measures its own baseline
  in-run rather than assuming that number.
- **Gate:** C6 (+5) re-run on TARGET first. It must
  reproduce 5/9 or the run STOPS — the setup changed
  and nothing else in the run would be comparable.
  (Same control-first reasoning as the hard probe.)

## Decision rule (frozen before the run)

For each alpha, let H = close+correct on HARD,
E = close+correct on TARGET, R = breakage regressions
(baseline passes, candidate fails) against the
in-run baseline.

- **EXTENDS** iff H >= 5/9 AND E >= 5/9 AND R = 0.
  Stronger forcing closes hard problems without
  losing easy accuracy. The generalisation claim
  extends to hard problems at that alpha.
- **TRADEOFF** iff H >= 5/9 but E < 5/9 or R > 0.
  It closes hard problems at the cost of easy ones.
  The claim does not extend cleanly; the numbers
  are reported as the tradeoff they are.
- **LIMIT** iff H < 5/9 for both alphas. Static
  forcing does not close hard problems. The forcing
  approach hits its limit, and the growing-bias
  variant becomes the next thing to scope.

Each alpha is judged on its own: one alpha may
extend while the other trades off, and that split
is itself the result.

First-appearance analysis (read-only, from saved
outputs, as in the hard probe): for the prompts
each alpha closes, did closure come before or after
derivation? This separates "stronger bias closes
too early" from "stronger bias closes after
derivation but the answer was still wrong".

## Setup (pinned, same as Stage 2 and both probes)

Qwen/Qwen3.5-0.8B, float16, Kaggle T4, thinking ON,
temp 0.6, top_p 0.95, top_k 20, seed 0, 1024-token
cap, presence_penalty attempted with the per-call
fallback. Raw outputs saved for every prompt.
Do NOT pip-upgrade transformers for this run.

## Steps

0. Re-resolve the `</think>` token id from a closed
   Sara generation (same method as Stage 2). Stop
   if no id resolves.
1. Baseline on HARD + TARGET + BREAKAGE (32 prompts),
   thinking ON.
2. Gate: C6 (+5) on TARGET. Must be 5/9 or stop.
3. C7a (+10) on all three batteries, with its own
   receipt.
4. C7b (+20) on all three batteries, with its own
   receipt.
5. First-appearance analysis per alpha.

## Cost and time

$0, Kaggle only. Runtime estimate roughly 60-90
minutes: baseline ~20 min (32 prompts), gate ~5 min,
each alpha ~20 min (32 prompts), plus derivation
and analysis. Well inside a 12-hour session. No
paid model API use, no new candidate families
beyond the two alphas.

## Risks and limits

- A larger alpha also changes the `</think>` input
  embedding more (tied head). Same disclosure as
  C6, larger magnitude — and one more reason the
  easy batteries are in the run: if the embedding
  change itself breaks easy prompts, E and R will
  show it.
- +20 may close so aggressively that easy problems
  close before derivation. That is what the E >= 5/9
  bar measures; a TRADEOFF verdict is a real result,
  not a failed run.
- This probe still tests static forcing only. If
  both alphas hit LIMIT, the next step is the
  growing-bias variant, scoped separately.
- The hard battery is still GSM-style arithmetic.
  An EXTENDS verdict extends the claim to harder
  arithmetic, not to all reasoning.
