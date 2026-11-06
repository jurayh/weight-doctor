# Weight Doctor Phase 2 — Stage 1 verdict (Nov 4, 2026)

Source: `phase2_stage1_results.json` (Tesla T4, torch
2.11.0+cu130, transformers 5.20.2, accelerate 1.15.0,
same sampling/seed as the probes). Baseline flags were
independently re-checked from the saved raw outputs:
0 mismatches across all 23 baseline prompts.

## Baseline: reproduced exactly

- Target: **0/9** (all unclosed at 1024 tokens).
- Known breakage: **4/4**, at exactly the same token
  counts as the probes (666, 350, 732, 134). The
  environment reproduced exactly.
- **New breakage: 4/10 pass, 6/10 fail.** The 6 failures
  (Japan, grass, pens, 10−4, Sara's apples, car) are
  all unclosed at 1024 tokens, not wrong answers.
  So the breakage base rate is **8/14**, not 14/14.
  Per the plan, breakage is a no-regression rule:
  a candidate must keep the 8 passing prompts passing;
  the 6 already-failing prompts cannot regress, and
  fixing some of them would be a bonus, not a
  requirement. This also means non-termination is
  broader than the target set: even trivial new
  prompts fail it.

## Diagnosis (a) marker rates: sustained, not decaying

- Maria: 22 markers total, 8 in the first 256-word
  block, 14 in the second. France: 34 markers, blocks
  8 / 19 / 7. Self-checking does not fade with length;
  it sustains (and for Maria rises) until the cap.
  This is not a tail that would end on its own.

## Diagnosis (b) </think> trajectory: INVALID, do not use

- Every sampled step, on all 3 prompts, records
  `think_logit = -inf` with `think_rank` 2–7.
  Those two numbers cannot both be true for raw
  logits: a −inf logit would rank near the bottom,
  not 2nd–7th. The recorded token id is 248069, yet
  the closed Sara run's final top token is 248044.
  Conclusion: the recorded id does not correspond
  to the token the model actually emits for
  `</think>`, or the returned scores are post-processed
  in a way the script did not account for.
- **The trajectory step must not be used to choose
  candidates.** Stage 2 must first resolve the correct
  `</think>` token id (encode it, and inspect the
  token id actually emitted at a closure) before any
  trajectory or lm_head-bias claim is trusted.

## Diagnosis (c) thinking-OFF: 9/9

- All 9 target prompts succeed with thinking OFF,
  in 10–207 tokens, with correct answers. The model
  can do the arithmetic and answer the controls; the
  failure is thinking-specific, not a capability gap.
  (Caveat: thinking-OFF rows save flags only, not
  full outputs, so they were not independently
  re-checked from raw text the way the baseline was.)

## Diagnosis (d) layer ablation: layer 15 stands out

Baseline markers: Maria 22, France 34, both unclosed
at 1024. Ablating one layer's MLP output (hook only,
weights unchanged):

| Layer | Maria | France |
|-------|-------|--------|
| 0 | closed, 283 tok, 2 markers | unclosed, 1024, 0 markers |
| 6 | unclosed, 1024, 6 | unclosed, 1024, 47 |
| 12 | unclosed, 1024, 16 | closed, 601, 10 |
| **15** | **closed, 428, 2** | **closed, 101, 0** |
| 19 | unclosed, 1024, 25 | unclosed, 1024, 70 |
| 23 | unclosed, 1024, 17 | unclosed, 1024, 22 |

- **Layer 15 is the only layer whose ablation closes
  both representative prompts, with marker collapse**
  (22→2, 34→0). Layer 0 closes Maria only, layer 12
  closes France only. Layers 6, 19, 23 close neither;
  layer 19 (one of Astra's layers) makes France worse
  (34→70 markers). Astra's 15/19 pair does not
  transfer as a pair here: 15 hits, 19 does not.
- **Caveat that matters:** ablation rows record
  closed and marker counts only, not correctness.
  Closed under ablation is not yet close+correct.
  Stage 2 must score any layer-15 candidate on the
  full battery with the close+correct rule before
  claiming anything.
- **Minimality caveat:** whole-layer MLP ablation is
  a diagnostic, not a passing edit. One layer's
  down_proj is ~3.6M weights, about 0.48% of the
  model — above Kill P2's 0.1% bar. A passing
  candidate must be column-limited within layer 15
  (or a direction / lm_head edit), not a whole-layer
  zeroing.

## Stage 1 verdict

Baseline reproduced, thinking-specific failure
confirmed, and a causal lead identified (layer 15).
The trajectory step failed its own consistency check
and is excluded from candidate selection. Stage 1
did not edit any weight.

## Proposed Stage 2 candidate order (frozen on go)

1. **Layer-15 column ablation.** First a column
   scan inside layer 15's MLP (intermediate-neuron
   groups, 2 prompts) to find which columns carry
   the self-check effect, then evaluate the
   column-limited ablation on the full 9+14 battery
   with its own receipt.
2. **Direction subtraction** at layer 15 (self-check
   direction from hidden states at self-check vs
   other tokens, scale grid 0.25 / 0.5 / 1.0), only
   if the column scan does not already pass P1.
3. **lm_head </think> bias**, only after the token
   id is resolved as in (b) above, and judged by
   close+correct (forced closure with a wrong answer
   is a failure, not a fix).
4. Whole-layer-15 ablation as a reference candidate
   only (expected P2 fail on size; it tests the
   mechanism, it is not a pass claim).

Budget stays at most 8 candidate evaluations, $0
Kaggle, each with a full-battery receipt, reported
against Kills P1–P4 including failures.
