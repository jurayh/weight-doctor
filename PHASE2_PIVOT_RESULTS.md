# Weight Doctor Phase 2 pivot — results

Scope (Kill P4): Qwen/Qwen3.5-0.8B, thinking ON, Kaggle
T4, temp 0.6 / top_p 0.95 / top_k 20, seed 0, 1024-token
cap, presence_penalty omitted (Kaggle transformers
rejects it). Target battery 9 prompts, breakage
battery 14 prompts. No claim about any other model,
mode, sampling, or battery.

## Verdict

**Kills P1 and P2 PASS, exactly at the bar, via one
candidate: C6 lm_head bias — 5/9 target, 0 breakage
regressions, 1,024 weights changed (0.00014%), single
lm_head row.** Kill P3 (LLM doctor) still UNTESTED.
Kill P4 holds: nothing here is a scale claim.

This is a pass, not a triumph. C6 is the "forces
closure" candidate: it does not remove the self-check
loop, it biases the model toward emitting `</think>`.
It passes by the narrowest margin the kills allow.

## How the numbers were earned

Three runs produced the same headline numbers:
- First run (Nov 4, console only, receipts lost):
  C1 5/9 with 2 regressions, C2 6/9 with 3, C6 5/9
  with 0; direction unevaluated (off-by-one bug).
- Crashed rerun (Nov 5, console only): C1, C2, and
  the scan reproduced, then died at C_direction 0.25
  on a float32/float16 dtype error.
- This run (Nov 5, full patched Stage 2, receipts
  in `kaggle/phase2-pivot/results-rerun/`): all 6
  candidates evaluated, no errors, no candidates
  skipped.

Baseline reproduced in every run: target 0/9, known
breakage 4/4 at the same token counts (666, 350, 732,
134), new breakage 4/10, so the breakage base rate
is 8/14. Baseline success flags were independently
re-checked from the saved raw outputs in this run:
0 mismatches across all 23 prompts. Every candidate
verdict was recomputed from its per-prompt rows
against the baseline: all 6 match their receipts,
and every receipt file matches the results JSON.

Column scan reproduced exactly across runs: best
group 5 (2240–2688, both closed, 3 markers), best
subgroup (2352–2464, both closed, 6 markers).
Token resolution confirmed `</think>` id 248069
from a closed generation, correcting the Stage 1
verdict's suggestion that the id was wrong — the
trajectory's −inf anomaly was about the scores, not
the id. Direction derivation worked after the dtype
fix: 17 self-check tokens vs 1,007 others, norm
1.128, dtype float16.

## Per-candidate results

| Candidate | Target | Regressions | P1 | P2 | Pass |
|-----------|--------|-------------|----|----|------|
| C1 subgroup ablation (112 cols, 114,688 w, 0.015%) | 5/9 | 2 (Sara, Earth) | No | Yes | No |
| C2 group ablation (448 cols, 458,752 w, 0.061%) | 6/9 | 3 (Sara, tables, January) | No | Yes | No |
| C_direction 0.25 (hook) | 3/9 | 3 (Sara, tables, garden) | No | Yes* | No |
| C_direction 0.5 (hook) | 5/9 | 1 (Sara) | No | Yes* | No |
| C_direction 1.0 (hook) | 3/9 | 2 (garden, Earth) | No | Yes* | No |
| **C6 lm_head bias (1 row, 1,024 w, 0.00014%)** | **5/9** | **0** | **Yes** | **Yes** | **Yes** |

\* Direction P2 passes on weight count only because
a hook changes no weights. That is not a surgical
weight fix, and none of the direction scales passed
P1 anyway, so the point is moot here.

Target passes, per candidate (close + correct):
- C1: Maria, Tom, Jake, notebooks, France.
- C2: Maria, Tom, rectangle, Jake, France, sky.
- Direction 0.25: Maria, notebooks, France.
- Direction 0.5: Maria, rectangle, Jake, France, sky.
- Direction 1.0: Tom, rectangle, sky.
- C6: Maria, baker, rectangle, notebooks, France
  (4 math + 1 control; fails Tom, train, Jake, sky).

Breakage, per candidate: every candidate also
*fixed* some prompts that failed at baseline (a
bonus, not a requirement). C6 fixed 3 of the 6
baseline-failing breakage prompts (Japan, 10−4,
Sara's apples) while preserving all 8 passing ones;
it still fails grass, pens, and car. C1 fixed the
same 3 while regressing Sara and Earth. Direction
0.5 fixed the same 3 while regressing only Sara —
the closest near-miss in the whole run.

## What the pattern says

- Ablation (C1/C2) closes more targets than any
  other family (C2 reaches 6/9) but breaks the most
  breakage prompts, including ones that closed in
  every probe (Sara under both, tables and January
  under C2). It is not selective enough.
- Direction subtraction peaks at scale 0.5 (5/9,
  1 regression) and falls off on both sides (3/9 at
  0.25 and 1.0). The direction exists — 17 self-check
  tokens carry a real mean difference — but
  subtracting it does not cleanly separate
  self-checking from everything else.
- The only passing candidate is the one that does
  not touch the loop at all: it makes `</think>`
  more likely. That is a real, receipted fix for
  this battery, and it is also the least
  mechanistic one. Do not present it as evidence
  that the self-check mechanism is understood.

## Disclosures

- C6's lm_head is **tied to the input embeddings**
  (`lm_head_tied_to_embeddings: true` in the
  receipt). The row edit therefore also changes the
  input embedding for the `</think>` token. The
  edit is still a single-row weight edit of 1,024
  weights, and P2 counts it as such, but the
  tied-embedding side effect is part of the edit.
- C6's alpha (0.0866) targets a +5 logit increase
  on average hidden alignment; the per-token
  increase varies.
- The direction candidates are hook interventions,
  not weight edits; their P2 is a weight-count
  technicality, disclosed in every receipt.
- No candidate was cherry-picked on one prompt:
  every candidate ran the full 9+14 battery, and
  every failure is reported above with its own
  receipt.

## Generalisation probe (Nov 5, 2026)

A held-out battery — 9 new target prompts + 8 new
breakage prompts, written and frozen before the run,
never used in candidate selection — was run to test
whether the Stage 2 pass generalises. Decision rule,
frozen before the run: C6 generalises if it scores
≥5/9 target with zero regressions on the held-out
battery; direction 0.5 is a comparator, judged by
the same bar. Results in
`kaggle/phase2-pivot/gen-probe-results/`.

**C6 generalises by the frozen rule: 5/9 target, 0
regressions on the held-out battery.** Direction 0.5
does not: 2/9, 0 regressions.

Held-out baseline: target 3/9 (Lily, Mia, pencils
close on their own), breakage 5/8 (the five short
prompts pass; the three word problems — Ben, chairs,
shelf — do not close). Both candidates preserve all
5 passing breakage prompts. C6 also fixes one
baseline-failing breakage prompt (Ben).

C6's held-out passes: Emma, Lily, garden, Mia,
pencils. Its failures: farmer, bus, Italy, banana.
Direction 0.5's passes: pencils, Italy.

Two honest caveats, both in the numbers:
- The net gain is smaller than on the frozen
  battery: +2 over a 3/9 baseline here, versus +5
  over a 0/9 baseline there. The absolute bar was
  the frozen rule and C6 meets it, but the effect
  is not as large on fresh prompts.
- C6 helps the math word problems and fails both
  one-sentence controls here (Italy, banana). On
  the frozen battery it passed France and failed
  sky; here it passes neither control. The bias
  generalises as a math-closure effect, not as a
  general closure effect.

Verification limits for this probe, stated plainly:
the probe's saved rows do not include raw outputs
(they were computed at generation time and then
dropped), so success flags cannot be independently
re-checked from text the way Stage 1/2 baselines
were. What was verified: every verdict recomputes
exactly from the saved per-prompt flags, every
receipt file matches the results JSON, token counts
are internally consistent (every closed prompt is
under the cap, every unclosed prompt is at it), and
the run reproduced the Stage 2 environment exactly
— direction derivation identical (17 self-check
tokens, norm 1.128) and C6's alpha identical
(0.0866).

## Hard-problem probe (Nov 12, 2026)

Question: does C6 generalise to problems where the
thinking does real work? Battery: 9 multi-step
problems, frozen Nov 7 in C6_HARD_PROBE_PLAN.md,
plus a control (C6 on the original 9 targets) and a
thinking-off diagnostic on the same 9 hard problems.
Decision rule frozen before the run: H = C6
close+correct on hard, T = thinking-off accuracy on
hard; extends iff H >= 5/9 and H >= T; hurts iff
H < T; partial otherwise. Results in
kaggle/phase2-pivot/hard-probe-results/.

- Baseline, thinking ON, hard battery: 0/9
  (non-termination, as expected).
- Thinking-OFF, hard battery: 7/9. The model can do
  these problems without thinking; its two failures
  are dropped-step errors (H5 forgets the 18 apples
  given away, H9 forgets the 14 baking eggs).
- C6 control (original targets): 5/9 — the setup
  gate passed, so the run is comparable.
- C6, hard battery: 0/9.

Verdict by the frozen rule: H = 0 < T = 7, so
FORCING CLOSURE HURTS, and the generalisation claim
narrows to easy problems.

The mechanism matters more than the label. C6 did
not fail by closing too early and producing wrong
answers. It failed by not closing at all: all 9 hard
prompts hit the 1024-token cap with the model still
second-guessing (the raw tails show it still
re-interpreting the problem at the cap). The bias
that closes easy thinking cannot overcome the
longer, more involved self-checking on hard
problems. And the model does derive the answer in
thinking on all 9 hard problems — the
first-appearance analysis finds the expected answer
in the unedited traces at tokens 336-922 — it just
never stops. So the narrowing is stronger than
"closes too early": on hard problems C6 does not
close at all. C6 is no worse than the unedited model
here (both 0/9 by non-termination), but much worse
than thinking-off (7/9): on hard problems, thinking
mode itself is the failure, and C6 does not rescue
it.

Verification, stated plainly: every stored flag was
recomputed independently from the saved raw outputs
and matches (this probe saved raw outputs, unlike
the generalisation probe); both receipts match the
results JSON exactly; the environment is identical
to the generalisation probe (torch 2.11.0,
transformers 5.20.2, accelerate 1.15.0) and C6's
alpha is identical (0.0866, tied). The first-
appearance token indices could not be re-derived
without the tokenizer (not installed locally), but
their consistency was checked from text: the
expected answer appears in the thinking portion of
every baseline trace, matching the stored
non-None first-appearance values, and every C6
close token is None exactly where the row is
unclosed.

What this leaves open: a stronger forcing (larger
alpha, or a bias that grows with thinking length)
is untested — but it would need its own plan, since
a larger alpha is a new candidate, not a re-run
of C6.

## What is still open

- Kill P3 (LLM doctor beats the scripted baseline):
  tested Nov 6 and FAILED as a tie (see KILL3_PLAN.md).
  Nothing in this run touches it.
- Whether a selective mechanism-level fix exists:
  no candidate in this run found one. Direction
  0.5, the closest near-miss on the frozen battery,
  fell to 2/9 on the held-out battery, so it is no
  longer the leading follow-up.
- C6's generalisation is now established for easy
  math problems on two batteries, and narrowed by
  the hard probe: on multi-step problems C6 does not
  close at all. Any claim beyond that (other models,
  other failure modes) is still untested.
