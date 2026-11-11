# C6 hard-problem probe — plan (pre-registered)

Written Nov 7, 2026, before any run. The question, the battery,
and the decision rule are fixed now; the build starts only on
Yuriy's go, and the run starts only after the probe is built.

## The question

C6 "generalises" on easy arithmetic, where the thinking is mostly
wasted second-guessing and forcing `</think>` early costs nothing.
That claim is now public. What was never tested: **problems where
the thinking does real work.** If the model needs the thinking to
derive the answer, C6's forced early closure should cut it off
before derivation and produce a wrong answer — and the
generalisation claim would need a narrower scope, stated in the
repo.

Why this is the crux: on the easy battery the model derives the
answer quickly (Maria's 32, the baker's 22 appear early in
thinking) and then second-guesses forever, so C6 closing at
~600–700 tokens comes *after* derivation. On a multi-step problem
derivation takes longer. C6's lm_head bias raises the `</think>`
probability uniformly — it does not wait for the answer. So the
timing of closure relative to derivation decides whether C6 helps
or hurts, and this probe measures exactly that.

## The battery (frozen in this plan)

Nine multi-step problems, each with a single-number answer,
verified by hand below. No repeat/count-again instructions
(the probe guard rejects those). None overlap the original 23 or
the held-out 17.

| # | Problem | Answer | Why it is hard |
|---|---------|-------:|----------------|
| H1 | A bakery bakes 240 loaves in the morning. They sell 3/8 of them before noon, then 145 in the afternoon. How many loaves are left at closing? | 5 | fraction, then two subtractions |
| H2 | A train travels 120 km in 2 hours, then 180 km in 3 hours. What is its average speed in km per hour? | 60 | total/total, not per-leg |
| H3 | Sara has 4 shelves with 18 books each. She gives away 25 books and buys 12 more. How many books does she have now? | 59 | three steps, mixed signs |
| H4 | Tom earns $15 per hour. He works 6 hours a day for 5 days, then spends $120 on groceries and $45 on a ticket. How much money does he have left? | 285 | rate × time, then two subtractions |
| H5 | A farmer has 96 apples. He packs 1/4 of them into boxes, gives 18 to a neighbour, and the rest are split equally between 2 baskets. How many apples are in each basket? | 27 | fraction, subtraction, division |
| H6 | A car drives 90 km in 1.5 hours, stops for 30 minutes, then drives 60 km in 1 hour. What is its average speed in km per hour over the whole trip, counting the stop? | 50 | hours/minutes conversion, stop counts |
| H7 | A shop sells pens in packs of 8 for $6 and packs of 12 for $8. Maria needs 24 pens. What is the cheapest total cost in dollars? | 16 | two options, compare |
| H8 | Lily reads 15 pages on Monday, twice as many on Tuesday as Monday, and 10 fewer on Wednesday than Tuesday. How many pages does she read in total? | 65 | chained dependencies |
| H9 | A farmer collects 5 dozen eggs each morning. He sells 3 dozen at the market and uses 14 eggs for baking. How many eggs are left? | 10 | dozen conversion, then subtractions |

Answer checks: H1 240×3/8=90, 240−90=150, 150−145=5. H2
(120+180)/(2+3)=60. H3 72−25+12=59. H4 450−120−45=285.
H5 24 packed, 72−18=54, 54/2=27. H6 150 km / 3 h=50.
H7 3×$6=$18 vs 2×$8=$16, min 16. H8 15+30+20=65.
H9 60−36−14=10.

Controls: the 9 original target prompts, run under C6 in the
same session. Expect ~5/9, reproducing the frozen-battery result.
If C6 scores below 3/9 there, stop — the setup changed, and the
hard-battery numbers would not be comparable.

Diagnostic: thinking-OFF accuracy on the same 9 hard problems
(10–200 tokens each). This tells us whether the model can do
them without thinking at all. If thinking-OFF is 0/9, the
"hurts" comparison is moot in that direction — the model cannot
do them anyway — and only the absolute bar carries weight.

## Setup (pinned, same as Stage 2 and the generalisation probe)

Qwen/Qwen3.5-0.8B, float16, Kaggle T4, thinking ON via chat
template, temp 0.6, top_p 0.95, top_k 20, seed 0 before each
generation, presence_penalty attempted with per-call fallback and
omission recorded, scoring cap 1024 tokens. C6 is the same edit
as Stage 2 (lm_head row += alpha × mean-final-hidden unit vector
from Sara, alpha targeting +5 average logit; tied-embedding
status recorded). Success = thinking closes (`</think>` appears
AND generation stops before the 1024 cap) AND the text after
`</think>` contains the expected answer (same checker as before:
numeric match on the standalone number at a word boundary).

One fix over the generalisation probe, built into this probe from
the start: **raw outputs are saved for every prompt**, so success
flags can be independently re-checked from text afterwards (the
earlier probe dropped them, and its flags could only be
recomputed from the flags themselves).

## Steps

1. **Baseline, thinking ON, hard battery** (9 prompts, 1024 cap).
   Rate measured in-run, not assumed. Expect 0/9 or near — the
   model does not terminate on easy prompts, and hard prompts
   should be no easier — but if it closes on any, that is data.
2. **Thinking-OFF diagnostic** on the 9 hard prompts.
3. **C6 on the hard battery** (9 prompts), same alpha method.
4. **C6 on the original 9 target prompts** (control).
5. **First-appearance analysis** (read-only, from saved outputs):
   in the unedited thinking traces, at what token does the
   expected answer first appear (as a standalone number), if at
   all — and does C6's closure token come before or after it?
   This is diagnostic, not a kill bar: it separates "C6 closed
   before derivation" from "C6 closed after derivation but the
   answer was still wrong".

## Decision rule (frozen before the run)

Let H = C6 close+correct on the hard battery, T = thinking-OFF
accuracy on the hard battery.

- **Generalisation extends to hard problems** iff H ≥ 5/9 AND
  H ≥ T. C6 closes thinking on hard problems without losing
  accuracy against not thinking at all.
- **Forcing closure hurts on hard problems** iff H < T. The
  generalisation claim narrows to easy problems, and the README
  must say so (the claim as published is too broad).
- **Partial** otherwise (T ≤ H < 5/9): C6 helps against
  non-termination but does not preserve full accuracy on hard
  problems. Reported as partial, with the numbers.

Whichever outcome, the verdict is appended to
`PHASE2_PIVOT_RESULTS.md` and the README's generalisation claim
is corrected to match it.

## Cost and time

$0, Kaggle only. Runtime estimate ~30–45 minutes: 9 hard
baseline + 9 hard C6 + 9 control C6 generations at the 1024 cap
(~50 s each at ~20 tok/s on reference kernels), plus thinking-OFF
runs (seconds each) and model load. No paid model API use, no
new candidate families, no new edits — C6 exactly as run before.

## Risks and limits

- If the control (original targets under C6) does not reproduce,
  stop: the setup changed and nothing else in the run is
  comparable.
- The hard battery is still GSM-style arithmetic, not a new
  failure mode. A pass here extends the claim to harder
  arithmetic, not to all reasoning.
- C6's closure timing varies by prompt; the first-appearance
  analysis is the check on whether a failure is a timing cut
  rather than a capability gap, but it is diagnostic only.
