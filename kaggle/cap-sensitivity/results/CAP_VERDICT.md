# Cap-sensitivity probe — verdict (Nov 4, 2026)

**Gate: PIVOT. Non-termination is real, not a cap artifact.**

Run: Kaggle free GPU, Tesla T4, torch 2.11.0+cu130,
transformers 5.20.2, accelerate 1.15.0, Qwen3.5-0.8B,
thinking on, temp 0.6 / top_p 0.95 / top_k 20,
presence_penalty omitted (same Kaggle rejection as the
loop hunt, recorded per prompt). First-run generation
1904 s (~32 min) for 13 prompts.

## Against the pre-registered gate

Gate set (the 9 Track B prompts that hit the 1024 cap
without closing): **9/9 still unclosed at 4096 tokens.**
Gate needed >= 5/9 unclosed to pivot. Result exceeds the
bar by the maximum possible margin: not one of the 9
closed even with four times the budget, including the two
trivial controls ("capital of France", "colour of the
sky"). Their 4096-token tails are still mid-self-check
("Wait, ...", "Okay.", "I'll write the response.") with
no </think> in sight.

Sanity set (the 4 prompts that already closed at 1024):
**4/4 closed, at exactly the same token counts as Track B**
(666, 350, 732, 134). No sanity failure, so the probe is
not incomplete: the environment reproduced Track B exactly
on prompts that terminate, and the gate set's failure to
terminate is a property of those prompts, not of the run.

## What this settles and what it does not

Settled: the 1024-cap reading was not a cap artifact.
Giving the model 4096 tokens on a trivial arithmetic
problem or a one-sentence trivia question does not make
it close thinking. The failure is thinking non-termination
on Qwen3.5-0.8B in thinking mode, under this sampling,
at $0 on a Kaggle T4.

Not settled, and not claimed here:
- Whether it would close at 8192 or 32768 tokens. That
  would be a further cap probe, not a refutation of this
  one: a model that needs >4096 tokens to answer
  "capital of France in one sentence" is already failing
  as an instruction follower, whatever happens later.
- Whether thinking OFF terminates cleanly (that would
  show the failure is thinking-specific, a useful doctor
  clue, but it is not part of this gate).
- Any edit, fix, or doctor design. This probe made no
  weight edit. Phase 2 pivot is now gated-open, not built.

## Phase 2 pivot, as gated-open

Target battery (proposed, needs its own go): the 9
gate prompts at the 1024 cap, success = closes thinking
AND gives the correct final answer. Base rate: 0/9.
Breakage battery (proposed): the 4 already-closed prompts
must still close with correct answers, plus short-QA and
general instruction tasks. Minimality and receipt rules
carry over from Phase 1 unchanged.
