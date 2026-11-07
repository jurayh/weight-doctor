# Kill 3 plan — LLM doctor vs scripted baseline (pre-registered)

Written Nov 5, 2026, before any paid call. The bar below is fixed
now; changing it after seeing the model's diffs would be moving
the goalposts.

## The question

Can an LLM, given exactly the weight evidence the scripted doctor
uses, diagnose the same four planted failures and propose surgical
diffs that fix them — and can it beat the scripted doctor at its
own game?

## What the LLM sees (fairness rule)

Per case, one prompt built by `weight_doctor/llm_doctor.py`:
the schema diagram plus the same weight evidence the scripted
doctor uses — W1 group norms (agent/goal/hazard), Wout row norms
and biases per action, Wout entries above the schema maximum (2).
It does NOT see episode outcomes, the catalog ground truth, or the
scripted doctor's diff. The full prompt for `dead_right` is 274
tokens; all four prompts are 274–298 tokens.

## Scripted baseline (measured, not assumed)

| Case | Weights changed | L2 |
|------|----------------:|---:|
| blind_goal | 18 | 4.24 |
| blind_hazard | 9 | 3.00 |
| dead_right | 13 | 7.42 |
| stuck_center | 1 | 10.00 |
| **Total** | **41** | max 10.00 |

## The bar — what "beat" means

Kill 3 **PASSES** iff the LLM doctor:

- fixes all 4 cases (every target episode passes, zero breakage
  regressions, judged by the same `verify_case` as the scripted
  doctor), AND
- changes strictly fewer than 41 weights in total across the 4
  cases, AND
- keeps every case's L2 at or under 20 (the Kill 2 bar).

A tie — exactly 41 weights, i.e. the identical repairs — is
**"matches, does not beat"**: Kill 3 FAILS, the LLM-doctor claim
goes, and the match is reported as a match. Fewer than 4 fixed,
any regression, or a total of 41 or more also fails.

Honest expectation, stated before the run: the scripted repair
restores schema values exactly, which is very likely the minimal
repair, so a strict beat may be unattainable and a tie is the
most likely good outcome. The run is worth doing anyway because
it measures whether an LLM can diagnose from weight evidence and
propose a surgical diff at all — a tie is a publishable result,
a failure to fix is a different, also publishable result.

## Call budget and model

- Model: `muse-spark-1.3-contributor` (contributor tier first,
  per the standing rule).
- Calls: 4 (one per case), plus at most 1 retry per case, and
  only when the response is not parseable JSON or the diff fails
  validation. Worst case 8 calls.
- Max output: 1024 tokens per call. Sampling is the API default
  (no temperature/seed parameter is passed); the run's outputs
  are receipted verbatim, so the verdict is about this run.
- Runner: `scripts/run_kill3.py`. A stub-client test
  (`tests/test_llm_doctor.py`, 5/5) already proves the
  prompt → parse → validate → apply → verify pipeline end to
  end for $0.

## Cost estimate (measured, Nov 5, 2026)

Measured with `python3 scripts/run_kill3.py --dry-run`: the four
exact prompt texts total **1,124 input tokens** (tiktoken cl100k).
That is a measurement of the real prompts, with one stated
caveat: cl100k is not the provider's own tokenizer, so the
provider's `usage` field is authoritative after the run, and
the cap is enforced against a 1.4x upper bound on top of it.

| Scenario | Input | Output | Contributor | Standard |
|----------|------:|-------:|------------:|---------:|
| Expected (4 calls, ~200 out each) | 1,124 | ~800 | ~$0.0003 | ~$0.005 |
| Worst case (8 calls, 1024 out each) | 2,248 | 8,192 | **$0.0019** | $0.0376 |

**Cap requested: $0.10** — about 50x the worst-case contributor
estimate, and still 2.6x the worst-case standard-tier price, so
even a full tier fallback plus every retry firing stays inside
it. The runner refuses any call that would cross the cap before
it is made, and reports measured provider tokens and dollars
afterwards next to this estimate.

No other paid work is covered by this approval: this is the Kill 3
run only, 4 cases, max 8 calls, one model, one cap.

## Smoke test (Nov 5, 2026) — pipeline failure, not a model verdict

Approved 1-case smoke test on `dead_right`, run Nov 5. Result:
**no valid diff, 0/4 fixed — but this is not a model failure.**

- Both calls (initial + 1 retry) returned 1024 output tokens
  and zero extractable text. The parser found no `output_text`
  parts, so the response could not even be parsed as JSON.
- Measured tokens vs estimate: provider reported 282 input
  tokens per call (cl100k measured 274 — close), 1024 output
  tokens each (the max-output cap, hit both times).
- Spent: **$0.0005** of the $0.10 cap, 2 calls. Saved in
  `reports/kill3_results.json`.
- Suspected cause: the client sent the request without
  `temperature` or a `reasoning` setting, so with default
  reasoning the model spent the whole output budget on reasoning
  items (which carry no `output_text`) before producing any
  answer text. The proven Sluice worker sets temperature 0 and
  reasoning effort low, and formats input as typed message items.
- Fix (not yet verified against the live API): `meta_client.py`
  now matches that request shape, and `extract_text()` raises
  with a truncated shape dump instead of silently returning an
  empty reply, so a repeat failure is diagnosable from the record.
  Re-running the smoke test needs a fresh approval: the approved
  call budget for this smoke test (1 + 1 retry) is spent.

## Smoke test re-run (Nov 5, 2026) — still no answer text

Re-run with the fixed client, same case, same $0.10 cap. Result:
**still no valid diff — and this time the response shape explains
why.**

- 1 call made. Provider usage: 282 input tokens, 1024 output
  tokens, of which **1022 are reasoning tokens**. Response
  status: `incomplete`, reason `max_output_tokens`. The output
  contains only a reasoning item (status incomplete, empty
  summary) and no answer text at all.
- So the first fix was not enough: even with reasoning effort
  low, the model spends ~1022 tokens reasoning on this task and
  the 1024-token output budget leaves nothing for the answer.
- Billing note: the runner recorded $0.00 / 0 calls for this run,
  which understates the real spend — the client tallied usage
  only after extracting text, so a no-text call tallied nothing.
  The provider did bill 282 in / 1024 out (~$0.0002). Fixed:
  the client now tallies usage immediately after receiving the
  payload, before extraction, and attaches usage to the raised
  error so the runner records it.
- Fix (not yet verified against the live API): raise
  `MAX_OUTPUT_TOKENS` from 1024 to **4096**. The output budget
  must cover reasoning (~1000 tokens observed) plus the answer
  (a JSON diff is a few hundred tokens). Recomputed worst case
  at 4096: 1-case smoke **$0.0017** contributor; full 8-call run
  **$0.0068** contributor ($0.1421 standard — above the cap, but
  the client enforces the cap in its own tier's rates, so spend
  can never cross $0.10 on either tier; on standard the run
  would stop early rather than overspend).
- Re-running needs a fresh approval again: this re-run's budget
  is spent.

## Smoke test, third run (Nov 5, 2026) — the model answers, the harness was wrong

Re-run with the 4096 output cap. Result: **the model now returns
JSON — and the failure is a harness bug, not a model failure.**

- 2 calls, spent **$0.0014** (564 in / 6,942 out tokens, measured
  from provider usage). Attempt 0 used 4,039 of the 4,096 output
  tokens — the reasoning is heavy on this task, so the 4096 cap
  is doing real work.
- Attempt 0 diagnosed correctly ("RIGHT action disabled: Wout
  RIGHT row zeroed with -5 bias, W1 groups healthy") and proposed
  `[["Wout", 0, 0, 1], ["bout", 0, 0, 0.0]]`. The validator
  rejected it: `bias entry for bout must have col null, got 0`.
- That is my bug, not the model's: the prompt said only
  `[[matrix, row, col, value], ...]` and never told the model
  that bias entries need `col` null. The model had no way to know.
  (Attempt 1, the retry, proposed only the bias fix with the same
  format issue, plus a wrong theory that the zero row was
  "default-action design" — retries on the same prompt are not
  guaranteed to improve.)
- Fix: the prompt now states the convention explicitly, with an
  example (`["bout", 0, null, 0.0]`), and a test pins it
  (`test_prompt_contains_evidence_and_schema`). The validator
  stays strict — a model that is told the rule and still writes
  0 fails on its own merits. The bar (fix 4/4, < 41 weights) is
  unchanged; only the missing format instruction was added.
- Prompt grew 274 → 351 input tokens for `dead_right`. Recomputed
  worst case: 1-case smoke $0.0017 contributor; full run $0.0068.
- Total paid spend on Kill 3 so far: ~$0.0021 across the three
  smoke runs ($0.0005 + ~$0.0002 + $0.0014), all inside the $0.10
  cap. Re-running needs a fresh approval again.

## Smoke test, fourth run (Nov 5, 2026) — runner crashed, reply lost

Re-run with the fixed prompt. The model's reply was received
and parsed — and then the runner crashed with
`KeyError: 'target_after'` while building the result row, before
anything was saved. **The model's diff for this run is lost:**
the results file still holds the previous run's rows.

- Cause: the runner read verifier keys (`target_after`,
  `breakage_after`) that `verify_case` does not return. The real
  keys are `target_after_fail` / `breakage_after_fail` (episode
  lists). The stub-client test never caught this because it
  tested `run_llm_doctor` + `verify_case` directly, not the
  runner's row building.
- Fix: `score_case()` in `scripts/run_kill3.py` now builds the
  row from the verifier's real keys, with a regression test
  (`test_score_case_row_uses_real_verifier_keys`) that scores the
  scripted diff for `dead_right` and asserts 195/195 target and
  309/309 breakage. Rows are now saved after every case, and the
  successful attempt's raw reply is recorded in full, so a later
  crash can no longer lose an already-paid-for reply.
- The paid call in this run was made and billed (amount unknown
  from the local record, since the crash happened before the
  tally was printed; the cap still bounds total spend). Total
  paid spend on Kill 3 is now ~$0.0021 plus this run's call.
- Re-running needs a fresh approval again. The prompt, the bar,
  and the cap are unchanged.

## Smoke test, fifth run (Nov 6, 2026) — first genuine model result

Re-run with the fixed runner. The smoke test finally completed
end to end: 1 call, **$0.0005** (360 in / 2,138 out tokens,
measured from provider usage), diff parsed, scored, and saved in
`reports/kill3_results.json`. And the result is a real model
result, not a harness bug.

- Diagnosis: correct. "Wout RIGHT output row ablated to zeros
  with bout suppressed to -5.0, disabling RIGHT action while W1
  is healthy." Format: correct, including `["bout", 0, null, 0.0]`.
- Diff: 10 weights — bout 0 → 0.0, and Wout row 0 cols 0–8 set to
  2.0 (cols 0–6) and 1.0 (cols 7–8). L2 7.42, coincidentally the
  same as the scripted repair's L2 (different distribution, same
  sum of squares).
- Verdict: **not fixed — target 95/195, breakage 297/309**
  (12 regressions). The model got the mechanism right (restore
  the RIGHT row) but the entries wrong: it guessed a contiguous
  block, cols 0–8, instead of the sparse schema pattern
  (cols 0, 4, 12, 16, 24, 28 at 2.0; cols 36, 40, 48, 52, 60, 64
  at 1.0).
- Why that is a fairness problem, not just a model miss: the
  evidence (row norms + biases) contains no information about
  which columns should be nonzero, and the prompt described the
  W1 wiring but never the Wout wiring. The scripted doctor gets
  its repair values from `build_schema_weights()`; the interface
  promises the LLM "the schema diagram". The prompt was missing
  half of it.
- Fix: the prompt now states the Wout wiring rule in full — unit
  c*4+d serves cell c in direction d, fallback units at 36+,
  Wout[d, c*4+d] = 2.0 and Wout[d, 36+c*4+d] = 1.0 wherever the
  neighbour exists, 0 elsewhere — plus the direction/cell layout
  needed to apply it. A test pins the rule text in the prompt.
  The bar is unchanged; the model is now given the same schema
  the scripted doctor uses, as the interface always promised.
- Prompt grew 351 → 586 input tokens for `dead_right`.
  Recomputed worst case: 1-case smoke $0.0018 contributor; full
  run $0.0070. Cap still $0.10.
- Total paid spend on Kill 3 so far: ~$0.004 across the five
  smoke runs, all inside the cap. Re-running needs a fresh
  approval again.

## Smoke test, sixth run (Nov 6, 2026) — fixed, exactly the scripted repair

Re-run with the full Wout wiring rule in the prompt. **Fixed:
195/195 target, 309/309 breakage, 13 weights, L2 7.42.**
1 call, $0.0004 (594 in / 1,808 out tokens, measured).

- The diff matches the scripted repair exactly, entry for entry
  (verified programmatically): the 12 sparse Wout entries plus
  bout 0 → 0.0. Given the full schema, the model derived the
  repair from the evidence and applied the wiring rule correctly.
- The smoke test has now done its job: the pipeline works end to
  end, the model can diagnose and repair when given the same
  schema the scripted doctor uses, and the measured cost per
  case is ~$0.0004.
- What is still untested: the other three cases, and the bar
  itself. `dead_right` at 13 weights ties the scripted repair on
  this case; Kill 3 needs all 4 fixed with a total strictly
  under 41, which means the LLM must find a strictly smaller
  repair somewhere — or tie everywhere and fail the kill as a
  match. The honest expectation from the plan still stands.
- Proposed next step (needs a fresh approval): the full 4-case
  run, all cases fresh in one run so the verdict lives in a
  single results file. Worst case $0.0070 contributor on the
  same $0.10 cap. Total paid spend on Kill 3 so far: ~$0.0044.

## Full run (Nov 6, 2026) — VERDICT: Kill 3 FAILS, as a tie

Approved full 4-case run, all cases fresh in one run. 4 calls,
no retries needed, **$0.0018** (2,405 in / 7,938 out tokens,
measured from provider usage), saved in
`reports/kill3_results.json`.

| Case | Fixed | Weights | L2 | Target | Breakage | Diff = scripted |
|------|-------|--------:|---:|-------:|---------:|:---------------:|
| blind_goal | yes | 18 | 4.24 | 444/444 | 60/60 | yes |
| blind_hazard | yes | 9 | 3.00 | 72/72 | 432/432 | yes |
| dead_right | yes | 13 | 7.42 | 195/195 | 309/309 | yes |
| stuck_center | yes | 1 | 10.00 | 102/102 | 402/402 | yes |
| **Total** | **4/4** | **41** | — | — | — | **all four** |

**Kill 3 FAILS.** The bar was "fix 4/4 with strictly fewer than
41 weights". The LLM fixed 4/4 with zero regressions — but with
exactly 41 weights, the identical repairs. Per the bar fixed
before the run, that is a tie: "matches, does not beat". The
LLM-doctor claim goes, as KILLS.md requires. The verdict is not
moved after the fact; the tie outcome was named in the plan
before any paid call.

What the tie means, plainly: the scripted repairs restore schema
values exactly, and on this toy those repairs are the unique
minimal ones — there is no strictly smaller correct repair for
the LLM to find. The bar was, in hindsight, near-unattainable,
and the run measured that rather than the model failing. The
substantive finding stands on its own: given the same weight
evidence plus the schema, the model diagnosed all four failures
correctly — including `stuck_center` down to the exact unit
("UP fallback unit 55 (cell 4 UP) inflated from schema 1.0 to
11.0") — and proposed the exact minimal repairs first try, with
no retries. That is a match, not a beat, and it is reported as
one.

Total paid spend on Kill 3: ~$0.005 across six smoke runs and
the full run, all inside the $0.10 cap. Nothing further is owed
on Kill 3: it is tested, failed as a tie, and closed.
