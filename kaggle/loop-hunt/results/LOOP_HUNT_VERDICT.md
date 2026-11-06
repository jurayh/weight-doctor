# Loop hunt — verdict (Nov 4, 2026)

**Verdict: NO CLEAN CASE. Phase 2 does NOT start on a loop
case, per the pre-registered decision rule. The negative
result stands; do not lower the bar.**

Run: Kaggle free GPU, Tesla T4, torch 2.11.0+cu130.
Raw results in this folder. Detector flags were
independently re-checked from the raw outputs: all 26
(hunt + controls, both tracks) match the script's flags.

## Against the pre-registered bars

Track A — Qwen2.5-0.5B-Instruct, greedy, 512 tokens:
0/10 hunt loops, 0/3 controls, 0 reproduced.
Bar needed >= 1/10, 0/3 controls, reproduced. FAIL.
Every hunt prompt stopped before the cap (212–449 tokens),
reruns identical. This model does not loop on these
open-ended prompts under greedy decoding.

Track B — Qwen3.5-0.8B, thinking on, 1024 tokens,
sampling temp 0.6 / top_p 0.95 / top_k 20, presence_penalty
OMITTED (Kaggle transformers rejected it; fallback recorded
in used_sampling and presence_penalty_used):
1/10 math loops, 2/3 controls looped, 1 reproduced.
Bar needed >= 4/10 math, 0/3 controls, reproduced. FAIL
on both the math rate and the control rate.
The one math loop, reproduced on a second seeded run, was
Maria's apples prompt, repeating the line "$24 - 7 = 17$."
The two looping controls were "capital of France" (repeating
"Paris is the capital of France.") and "colour of the sky"
(repeating "Okay."). A control that loops is the opposite of
a clean case: the loop is not math-specific here.

## What Track B actually showed (not a verbatim loop)

7/10 math prompts hit the 1024-token cap, but only 1 was a
verbatim loop under detector v2. The other cap-hitters, and
the looping controls, show a different failure: thinking
that never terminates. The model works out the right answer
in its thinking (Maria 32, baker 22, Tom 252, rectangle 36
all appear in the thinking text), then keeps second-guessing
— "Wait, ...", "Okay.", "Final Answer: X" followed by
another check — and never emits </think>. Only 4/13 Track B
prompts closed thinking and gave a final answer (Sara 47,
tables 8, garden 46, and the 2+3 control giving 5, all
correct; correction Nov 4: an earlier count said 3/13 by
omitting the 2+3 control). That is a
non-termination / thinking-closure failure, not the verbatim
repetition loop the probe was pre-registered to find, and
it must not be relabelled as a loop case after the fact.

Caveats that cut both ways, recorded, not hidden:
- Sampling deviation: without presence_penalty (which
  suppresses repetition), this run was MORE loop-prone than
  Astra's exact protocol, yet still only 1/10 verbatim loops.
- Prompt difficulty: these were trivial arithmetic problems,
  not MATH/GSM-style problems, and the cap was 1024, not
  Astra's 32768. A harder battery at a longer cap could
  behave differently; that would be a new probe with its own
  pre-registered bar, not a reinterpretation of this one.
- Track B structure (dump only, no edit): 24 layers,
  model.layers, total params 752,393,024 as reported,
  layers 15 and 19 MLP down_proj shapes [1024, 3584].

## What this does and does not close

Closed: Phase 2 on a verbatim loop case from this battery,
on either model. The toy Phase 1 result stands untouched.
Not closed: Phase 2 on a different failure family (e.g.
thinking non-termination, with its own battery), or a new
Astra-faithful probe (harder problems, longer cap,
presence_penalty supported). Neither is approved by this
verdict; each needs its own go.
