# Kill criteria (pre-registered, before the catalog run)

These bars were fixed before `scripts/run_catalog.py` was run.
Changing them after seeing results would be moving the goalposts.

## Kill 1 — fixes with zero regressions

The scripted doctor must fix at least 3 of the 4 planted failures.
A case counts as fixed only if, after the edit:

- every target-battery episode passes (the failing behaviour now
  works, across all target episodes, not one lucky run), AND
- zero breakage-battery episodes regress (no episode that passed
  before the edit fails after it).

A fix that breaks something else is not a fix.

## Kill 2 — edits stay surgical

For every fixed case:

- fraction of weights changed <= 2%, AND
- edit L2 norm <= 20.0.

An edit that rewrites the network does not count as a fix; it is
retraining in disguise. (For scale: the toy has 20,612 parameters,
so 2% is 412 weights. The repairs here touch 1-18 weights.)

These bars are also enforced in code (`verifier.py`).

## Kill 3 — LLM doctor must beat the scripted baseline, or the claim goes

NOT RUN in the $0 core. The LLM-doctor interface is defined in
`weight_doctor/llm_doctor.py`, but running it against a paid API
needs an estimate-first approval under the standing paid-model
rule, and no such approval has been given. Until that run happens,
kill 3 is UNTESTED: the harness stands, the LLM-doctor claim does
not. It is not passed, and it is not failed; it is owed.

## Kill 4 — no scale claims leave the toy room

The README and every receipt label results as toy results
(3x3 grid MLP, 20,612 parameters). Small-model results, when Phase 2
runs, will be labeled as small-model results in the same way.
Any document in this repo that claims a toy result generalises to
a larger model fails this kill.

## Phase 2 gate

Phase 2 (a Qwen-class 0.5B/0.8B model with a known failure, loop-bug
family first) starts only if kills 1, 2, and 4 pass, AND the compute
path is settled first: a 0.8B model runs locally but slowly on CPU,
and the RTX 4090 laptop path is unverified (OS and VRAM
unconfirmed), so where the model actually runs must be confirmed
rather than assumed. Kill 3 remaining untested does not block the
harness, but it does block any LLM-doctor claim in Phase 2.
