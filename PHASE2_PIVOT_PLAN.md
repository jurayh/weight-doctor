# Phase 2 pivot plan — thinking non-termination in Qwen3.5-0.8B

**Status: PLAN ONLY, awaiting a build go. No code, no edit, no
Kaggle run beyond the probes already completed.**

This plan replaces the loop-bug target, which failed its
pre-registered bars on Nov 4, 2026 (loop hunt: Track B 1/10
math loops, 2/3 controls looped). It does NOT reinterpret
that failure. The pivot target is a different, separately
measured failure: thinking that never terminates.

## 0. The failure, with its evidence

Qwen/Qwen3.5-0.8B, thinking mode ON, Kaggle T4, sampling
temp 0.6 / top_p 0.95 / top_k 20 (presence_penalty omitted
on Kaggle, recorded), seed 0:

- At a 1024-token cap (loop hunt Track B): 4/13 prompts
  closed thinking, 9/13 hit the cap still thinking, even
  though the right answer was already in the thinking text.
- At a 4096-token cap (cap-sensitivity probe, same
  model/sampling/seed, only the cap changed): 9/9 gate
  prompts still unclosed, 4/4 sanity prompts closed at
  exactly the same token counts as before. Gate PIVOT
  (>= 5/9 unclosed) passed by the maximum margin.
- The failure is therefore not a cap artifact, not a
  verbatim repetition loop (only 1/10 math prompts looped
  verbatim), and not a knowledge failure (thinking
  contains the right answers). It is: the model cannot
  stop checking itself and emit </think> + a final answer.

## 1. Pinned setup (nothing changes mid-battery)

- Model: Qwen/Qwen3.5-0.8B, float16, Kaggle T4, $0.
- Mode: thinking ON, via the chat template
  (enable_thinking=True, with the loop-hunt fallback).
- Sampling: temp 0.6, top_p 0.95, top_k 20, seed 0 before
  each generation; presence_penalty attempted with the
  per-call fallback, omission recorded. If a later
  environment supports presence_penalty, that is a NEW
  battery, not a silent change to this one.
- Cap for all target/breakage scoring: 1024 tokens.
  (4096 was the gate probe's cap; the target battery
  scores at 1024, where the failure is frequent and
  cheap to measure.)
- Versions, GPU name, and sampling actually used are
  logged in every run's output, as in the probes.

## 2. Batteries (frozen before any edit is tested)

### Target battery — 9 prompts, base rate 0/9

The 9 cap-sensitivity gate prompts, verbatim:

| # | Prompt | Expected final answer |
|---|--------|----------------------|
| 1 | Maria has 24 apples. She gives 7 to her brother and buys 15 more. How many apples does she have now? | 32 |
| 2 | A baker makes 5 trays with 12 muffins each. He sells 38 muffins. How many muffins are left? | 22 |
| 3 | Tom reads 18 pages a day. How many pages does he read in 2 weeks? | 252 |
| 4 | A rectangle is 9 cm long and 4 cm wide. What is its area in square cm? | 36 |
| 5 | A train travels 60 km per hour for 3 hours, then travels 40 km more. How far did it travel in total, in km? | 220 |
| 6 | Jake earns $8 per hour and works 6 hours a day, 5 days a week. How much does he earn in a week, in dollars? | 240 |
| 7 | A store sells notebooks for $3 each. Ana buys 7 notebooks and pays with $25. How much change does she get, in dollars? | 4 |
| 8 | What is the capital of France? Answer in one sentence. | Paris |
| 9 | What colour is the sky on a clear day? Answer in one sentence. | blue |

Success on a prompt = BOTH: (a) thinking closes
(</think> appears AND generation stops before the
1024 cap), AND (b) the text after </think> contains the
expected answer (number, or "Paris"/"blue",
case-insensitive). Closing with a wrong answer is NOT a
success — that is the guard against an edit that just
forces closure.

### Breakage battery — must all still pass after any edit

- The 4 already-closed prompts, verbatim (Sara 47,
  tables 8, garden 46, 2+3 = 5), same success definition
  (close + correct). Base rate 4/4.
- 10 NEW short-QA / short-instruction prompts, written
  at build time and FROZEN before the first candidate
  edit is evaluated (same style: one fact, one short
  calculation, one instruction with a constraint; none
  asking for repetition). Success = close + correct.
  Writing them at build time is fine; changing them
  after seeing edit results is not. Their baseline
  pass rate is measured in Step 1 and is NOT assumed
  to be 10/10: breakage means no prompt that passed
  at baseline regresses, not that every new prompt
  must pass (see Kill P1).
- Breakage is reported per prompt, not just as a total,
  so a single regression cannot hide inside an average.

### What is NOT in the batteries (scope control)

- No thinking-OFF prompts in the scored batteries.
  Thinking-OFF is a diagnostic clue (see §3), not a
  scored target: scoring it would change the failure
  being fixed.
- No MATH/GSM test-set problems. The battery is 9
  trivial prompts on purpose: if the model cannot close
  on these, harder problems add noise, not signal.
- No accuracy-only metric. A model that closes with
  wrong answers has not been fixed; a model that
  answers correctly inside unclosed thinking has not
  been fixed either. Both conditions are required.

## 3. Doctor (scripted first, as in Phase 1)

The doctor is scripted and hypothesis-driven. It reads
weights and activations, proposes diffs only, and never
rewrites the model. No gradient training, no fine-tuning,
no RL. The LLM doctor stays UNTESTED (see Kill 3).

### Step 1 — baseline re-run (read-only)

Re-run target + breakage on the base model in the build
notebook, to confirm base rates (0/9 target, 4/4 known
breakage, and whatever the 10 new breakage prompts
score) in the build environment before any edit.
If the target/known-breakage baseline does not
reproduce, the build stops and the discrepancy is
reported; edits are not evaluated against a baseline
that does not hold.

### Step 2 — diagnosis (read-only, no edit)

All of these are measurements, in order, each with its
output recorded before any candidate edit is chosen:

1. **Closure markers over time.** For closed vs
   unclosed prompts, count self-check markers per
   256-token block ("Wait", "Okay", "Final Answer",
   "check", "confirm") and track whether their rate
   rises, falls, or stays flat as thinking goes on.
   Question: is unclosed thinking a rising self-check
   loop, or a flat refusal to commit?
2. **</think> logit trajectory.** At each generation
   step (sampled, not every token, for cost), record the
   rank and probability of the </think> token and of
   EOS. Question: does the model ever seriously consider
   closing, or is </think> never in contention?
3. **Thinking-OFF comparison (diagnostic only).** Run
   the 9 target prompts with thinking OFF. If they
   terminate cleanly, the failure is thinking-specific;
   if they do not, the failure is broader than thinking.
   Either result is recorded and shapes the candidate
   edits; neither changes the scored batteries.
4. **Layer scan.** For a small set of layers (including,
   but not starting from, layers 15 and 19 — Astra's
   layers are a hypothesis to test, not a starting
   point), measure which layers' MLP outputs most
   strongly predict a self-check token next. Method:
   activation correlation + targeted ablation (zero one
   layer's MLP output on a prompt, see if self-check
   rate moves). This is the doctor's own measurement;
   copying Astra's layer/column choice without it would
   be copying, not diagnosing.

### Step 3 — candidate edits (each as a diff, each fully scored)

A pre-registered candidate family list, evaluated in
order, each on the FULL target + breakage batteries
(never on one prompt first, never cherry-picked):

1. **Direction subtraction.** Subtract a scaled
   self-check direction (derived from Step 2.4, not
   from Astra) from the MLP output of the layers it
   implicates. Scale swept over a small fixed grid
   (e.g. 0.25, 0.5, 1.0), each scale a separate
   candidate with its own receipt.
2. **Targeted ablation.** Zero or scale down the
   specific MLP columns / attention heads that Step 2.4
   implicates, one candidate per layer set.
3. **</think> logit bias via lm_head.** Add a small
   constant to the lm_head row for the </think> token
   (a weight edit, counted in minimality, disclosed in
   the receipt). This is the "forces closure" candidate:
   it is expected to close more prompts, and the
   close+correct success definition is what decides
   whether that closure is a fix or a cheat.
4. **Stop.** If no candidate passes Kill 1, the build
   reports that honestly and stops. No new candidate
   families are invented after seeing the results
   without a new plan approval.

At most 8 candidate evaluations are budgeted (see §5).
Each candidate's diff, target per-prompt results,
breakage per-prompt results, weight fraction, and L2
are emitted as a receipt (JSON + Markdown), exactly as
in Phase 1.

## 4. Pre-registered kills (Phase 2 pivot)

Fixed now, before any build code runs. Changing them
after seeing edit results would be moving the goalposts.

- **Kill P1 — fixes with zero regressions.** At least
  **5 of 9** target prompts succeed (close + correct),
  AND **zero** breakage prompts regress (every
  breakage prompt that passed before still passes
  after, close + correct). A fix that breaks something
  else is not a fix. (Bar set at a majority, not at
  9/9: 0/9 to 5/9 with zero regressions is already a
  strong, honest claim; 9/9 would be a bonus, not the
  bar.)
- **Kill P2 — edits stay surgical.** For every passing
  candidate: fraction of weights changed <= 0.1%
  (of ~752M reported parameters, so <= ~752k weights),
  AND the edit is column/head/layer-limited (not a
  full-matrix rewrite). L2 is reported in every receipt
  but has no absolute bar, because absolute L2 scales
  with model size and the toy bar (<= 20) does not
  transfer. An edit that rewrites the network does not
  count, whatever its L2.
- **Kill P3 — LLM doctor, still untested.** Carried over
  from Phase 1, unchanged: the LLM-doctor claim does not
  stand until an LLM doctor beats the scripted baseline
  on this battery, with an estimate-first paid-API
  approval. This build does NOT attempt it. No paid
  model API use is approved by this plan.
- **Kill P4 — no scale claims leave this room.** Every
  result is labeled: Qwen3.5-0.8B, thinking mode ON,
  Kaggle T4, this sampling, this 9+14 battery. No claim
  about larger models, other Qwen versions, thinking
  OFF, or "fixing LLM loops" in general. Any document
  that generalises beyond this battery fails this kill.

## 5. Cost, time, and deliverables

- **Cost: $0.** Kaggle free tier only. No rental, no
  paid API. Weekly Kaggle quota (~30 GPU-hours) is the
  binding constraint, not money.
- **Time (measured, not guessed):** Track B first-run
  was ~547 s for 13 prompts at 1024 tokens (~20 tok/s
  on reference kernels). A full battery run here is
  23 prompts (9 target + 14 breakage) at 1024 tokens:
  roughly 15–20 minutes per candidate evaluation,
  plus model download once per session. Budget: 1
  baseline run + up to 8 candidate runs = roughly
  2.5–3.5 hours of GPU time, splittable across
  sessions (results saved after every prompt, as in
  the probes). If a session ends mid-candidate, that
  candidate's partial results are NOT counted; it is
  re-run whole.
- **Deliverables:**
  - `weight_doctor/phase2/` — doctor, verifier,
    receipt code for this battery (new code for a
    real model; toy code is not reused blindly).
  - `kaggle/phase2-pivot/` — notebook + script +
    README + results + receipts for the Kaggle runs.
  - `PHASE2_PIVOT_RESULTS.md` — per-candidate results
    against Kills P1–P4, including failures, with the
    same honesty standard as Phase 1 (a failed
    candidate is a result, not a deleted row).
- **Stages, as usual:** this plan (approval) ->
  build (doctor + frozen batteries) -> run candidates
  -> report vs kills. A go on this plan approves the
  build only; a go on the build does not pre-approve
  any claim beyond what Kills P1–P4 actually measure.

## 6. Risks and failure modes, stated in advance

- **Forced closure with wrong answers.** Guarded by
  the close+correct success definition. If Kill P1 is
  met only by the lm_head-bias candidate and its
  answers are wrong, it fails P1 by definition.
- **Math-only or control-only fixes.** Guarded by
  per-prompt reporting: a candidate that fixes 5 math
  prompts but breaks both controls (or vice versa)
  fails P1 on breakage, and is reported as such.
- **Seed / hardware variance.** Seeded runs on the
  same GPU type; versions and GPU name logged. If the
  baseline re-run does not reproduce 0/9 and 4/4, the
  build stops (see §3 Step 1) instead of proceeding
  on a shifting baseline.
- **Reference-kernel slowness.** Track B and the cap
  probe already ran on reference kernels; time
  estimates above use that measured speed, not a
  native-kernel speed.
- **The edit works here and nowhere else.** That is
  exactly what Kill P4 forbids claiming otherwise.
  The result, if it passes, is "this doctor fixed
  thinking non-termination on this model, this
  sampling, this battery" — and that is a real,
  checkable result, not a general one.

## 7. Out of scope (unchanged from the brief)

Gradient-based training / fine-tuning, RL on weight
editing, large-model editing claims, safety-critical
edits, copying Astra's edit, loading the third-party
rebuilt Astra-edited checkpoint, changing sampling or
batteries mid-run, and any paid model API use without
a separate estimate-first approval.
