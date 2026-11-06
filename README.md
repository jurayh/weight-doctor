# Weight Doctor

An open harness for diagnosing and fixing model behaviour through
weight edits, where every edit ships with a verification receipt.
The edit techniques are established — steering, ablation, surgical
weight changes, ROME lineage — and this project says so. The new
part is the loop:

**diagnose → edit → verify the fix → verify nothing else broke →
emit the evidence.**

Out of scope for v1: gradient-based training or fine-tuning (edits
only), RL on weight editing, large-model editing claims, and
safety-critical edits. The LLM doctor is untested (see below).

## What is in this repo

- **Phase 1 — toy harness** (`weight_doctor/`, `tests/`,
  `scripts/`, `receipts/`, `reports/`): a 3×3 grid game with a
  small ReLU MLP policy, a planted-failure catalog, a scripted
  doctor, a verifier, and a receipt per case. Runs on local CPU,
  $0, no model APIs.
- **Phase 2 — small real model** (`kaggle/`): probes and candidate
  evaluations on Qwen3.5-0.8B in thinking mode, run on Kaggle's
  free GPU. The model work lives in self-contained notebooks;
  the torch-free harness logic lives in `weight_doctor/phase2/`.
- **Plans and results**: [`PHASE2_PIVOT_PLAN.md`](PHASE2_PIVOT_PLAN.md)
  (pre-registered design and kills), [`KILLS.md`](KILLS.md)
  (Phase 1 kills), [`PHASE2_PIVOT_RESULTS.md`](PHASE2_PIVOT_RESULTS.md)
  (Phase 2 verdict, including the generalisation probe).

Nothing in this repo is a claim about any model other than the one
named next to each result: the toy MLP in Phase 1, Qwen3.5-0.8B in
thinking mode in Phase 2.

## One worked case, end to end

The toy is a 3×3 grid. Cells are numbered row-major: 0=(0,0), 1=(1,0),
2=(2,0), 3=(0,1), and so on. A policy is a ReLU MLP, 27 inputs →
128 → 128 → 4 outputs, 20,612 parameters, built from a published
wiring schema (`weight_doctor/policy.py`): agent/goal/hazard
pass-through units, safe-reducing units, and a hazard-detour
fallback that only fires when the goal channel is live.

**The failing policy.** Case `dead_right`: the RIGHT output row is
zeroed and its bias set to −5, so RIGHT can never win argmax.

Take episode (agent 0, goal 1, hazard 2): start at (0,0), goal one
step east at (1,0), hazard at (2,0).

- Broken policy: DOWN to (0,1), then UP back to (0,0), then DOWN
  again — a loop. It never reaches the goal, because the only move
  that helps is the dead one.

**The diagnosis.** The scripted doctor reads the weights by layer
and channel (`weight_doctor/doctor.py`) and reports:

> Output row for RIGHT is zero and its bias is −5, so RIGHT can
> never win argmax.

That matches the catalog ground truth exactly.

**The diff.** The doctor proposes a diff, not a rewritten model:
restore the RIGHT output row to its schema values (12 weights) and
its bias to 0 (1 weight). 13 weights changed out of 20,612
(0.0631%), edit L2 norm 7.42. Full entry list in
[`receipts/dead_right.json`](receipts/dead_right.json).

**The batteries.**

- Target battery (the failing behaviour): 195 episodes the healthy
  policy passes and the broken policy fails. Before the edit: 0
  passing. After the edit: **195/195 passing**.
- Breakage battery (everything else): 309 episodes the broken
  policy already passed. After the edit: **309/309 still passing,
  0 regressions**.

**The receipt.** [`receipts/dead_right.md`](receipts/dead_right.md)
records the diagnosis, the diff, both batteries before/after, and
the broken/edited weight hashes.

**Edited policy, after the fix:** from (0,0) it goes RIGHT, directly
to the goal. Red to green, nothing else moved.

## Toy catalog results

504 episodes total, enumerated completely — no sampling. The
healthy policy passes all 504. Target = healthy-pass minus
broken-pass; breakage = healthy-pass intersect broken-pass.

| Case | Target | Breakage | Weights changed | Fraction | L2 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| blind_goal (ignores goal channel) | 444/444 | 60/60 | 18 | 0.0873% | 4.24 | FIXED |
| blind_hazard (ignores hazard channel) | 72/72 | 432/432 | 9 | 0.0437% | 3.00 | FIXED |
| dead_right (dead RIGHT action) | 195/195 | 309/309 | 13 | 0.0631% | 7.42 | FIXED |
| stuck_center (local minimum at centre) | 102/102 | 402/402 | 1 | 0.0049% | 10.00 | FIXED |

Full numbers in [`reports/catalog_results.json`](reports/catalog_results.json);
one receipt pair per case in [`receipts/`](receipts/).

## Verdict against the kill criteria

Pre-registered bars live in [`KILLS.md`](KILLS.md) and are enforced
in code (`weight_doctor/verifier.py`).

- **Kill 1 (fix ≥3/4 with zero regressions): PASS — 4/4 fixed.**
- **Kill 2 (surgical: ≤2% of weights, L2 ≤20): PASS** — largest
  edit is 18 weights (0.0873%) and the largest L2 is 10.00.
- **Kill 3 (LLM doctor beats scripted baseline): UNTESTED.** The
  interface is defined in `weight_doctor/llm_doctor.py`, but running
  it against a paid API needs an estimate-first approval, which has
  not been given. The harness stands; the LLM-doctor claim does not.
  It is not passed and not failed; it is owed.
- **Kill 4 (no scale claims leave the toy room): PASS** — the
  Phase 1 receipts are labeled toy-only, and the Phase 2 results
  are labeled Qwen3.5-0.8B only (see the Phase 2 section above).

## Honest limitations (read before citing this)

1. **The toy doctor is schema-guided.** It checks weights against
   the published wiring diagram and repairs deviations. That is the
   toy analogue of diagnosis, not a demonstration that a doctor can
   diagnose an arbitrary trained model from weights alone.
2. **Edited weights coincide with healthy weights.** Because the
   schema values are exact, a successful repair restores the healthy
   policy exactly (same hash across all four cases). That is a
   property of the toy, not evidence that real repairs are exact.
3. **The toy game is fully enumerable.** Real behaviours are not.
   The drift-gate shape of the breakage battery (before/after on a
   fixed set) is what transfers, not the 504-episode completeness.

## Phase 2: a real small model, and what it actually showed

The loop-hunt found no clean verbatim-loop case (Qwen2.5-0.5B:
0/10; Qwen3.5-0.8B thinking: 1/10, with 2/3 controls looping),
so Phase 2 pivoted to what the data actually showed: **thinking
that never terminates**. At a 4096-token cap, 9/9 gate prompts were
still unclosed, including both trivial controls, while 4 sanity
prompts closed at exactly the same token counts as before — so
non-termination is real, not a cap artifact.

The pivot's target: 9 gate prompts at a 1024-token cap, success =
thinking closes *and* the final answer is correct. Breakage: 4
prompts that already closed, plus 10 new short prompts; the
breakage base rate turned out to be 8/14, and breakage is a
no-regression rule on those 8.

Diagnosis: thinking-off succeeds 9/9 (the failure is
thinking-specific, not a capability gap), self-check markers
sustain rather than decay, and layer-15 MLP ablation closes both
representative prompts while layer 19 makes one worse. The
`</think>` logit trajectory from Stage 1 was thrown out as
invalid (a −inf logit recorded with rank 2–7 cannot both be true)
and played no part in candidate selection; the token id itself
(248069) was later confirmed from a closed generation.

Candidates, each on the full 9+14 battery with its own receipt:

| Candidate | Target | Regressions | Verdict |
|-----------|--------|-------------|---------|
| C1 subgroup ablation (112 cols, 0.015%) | 5/9 | 2 | Fail P1 |
| C2 group ablation (448 cols, 0.061%) | 6/9 | 3 | Fail P1 |
| Direction 0.25 (hook) | 3/9 | 3 | Fail P1 |
| Direction 0.5 (hook) | 5/9 | 1 | Fail P1 |
| Direction 1.0 (hook) | 3/9 | 2 | Fail P1 |
| **C6 lm_head bias (1 row, 1,024 w, 0.00014%)** | **5/9** | **0** | **Pass P1+P2** |

**Generalisation probe.** C6 and direction 0.5 were re-tested on a
held-out battery (9 new target + 8 new breakage prompts, frozen
before the run). C6 scored **5/9 with 0 regressions — it
generalises by the frozen rule**; direction 0.5 scored 2/9 and
does not. Caveats, in the numbers: the held-out baseline was
already 3/9, so the net gain is +2 (versus +5 over a 0/9 baseline
on the frozen battery), and C6 fails both one-sentence controls
on the held-out battery. It generalises as a math-closure effect,
not a general closure effect.

**What C6 is, plainly.** It is the "forces closure" candidate: it
biases the model toward emitting `</think>`. It does not remove
the self-check loop, and it passes exactly at the bar. Its lm_head
is tied to the input embeddings, so the row edit also changes the
input embedding for the `</think>` token — that is disclosed in
the receipt. Do not present this as evidence that the self-check
mechanism is understood. No selective, mechanism-level fix was
found in this run.

**Kills.** P1 (≥5/9, zero regressions) and P2 (≤0.1% of weights,
layer-limited) pass via C6, exactly at the bar. P3 (LLM doctor
beats the scripted baseline) is **untested** — the interface exists
(`weight_doctor/llm_doctor.py`), no paid model API use has been
approved or run. P4 (no scale claims) holds: every result here is
labelled Qwen3.5-0.8B, thinking mode, Kaggle T4, this sampling,
this battery. Full detail in
[`PHASE2_PIVOT_RESULTS.md`](PHASE2_PIVOT_RESULTS.md).

## Run it

```bash
python3 tests/test_weight_doctor.py   # 6/6, stdlib only, no pytest needed
python3 scripts/run_catalog.py        # full catalog + receipts, local CPU, $0
```

The Kaggle notebooks are self-contained (one code cell each) under
`kaggle/`: smoke test, loop hunt, cap-sensitivity probe, Stage 1
baseline + diagnosis, Stage 2 candidates, and the generalisation
probe. Each folder has a README with run steps. The pattern that
matters, learned the hard way twice: **save the output files
before the Kaggle session ends** — `/kaggle/working/` is
per-session only, and two runs lost their files to that. Results
already obtained are committed under `kaggle/*/results/` with
verdict files.

Dependencies: Python 3.12, numpy only. No network, no model APIs,
no training. The doctor edits weights only; nothing here trains or
fine-tunes.

## Components

- `weight_doctor/game.py` — 3×3 grid, 504 enumerable episodes.
- `weight_doctor/policy.py` — MLP, wiring schema, diff application,
  minimality stats, weight hashes.
- `weight_doctor/catalog.py` — planted failures with ground truth;
  target/breakage batteries computed, not hand-picked.
- `weight_doctor/doctor.py` — scripted doctor: read weights by
  layer/channel, name the cause, propose a diff.
- `weight_doctor/verifier.py` — target battery, breakage battery,
  minimality bars.
- `weight_doctor/receipt.py` — one document per edit.
- `weight_doctor/llm_doctor.py` — LLM-doctor interface, defined
  but not run (kill 3 untested).
- `weight_doctor/phase2/` — torch-free Phase 2 harness logic
  (frozen batteries, verifier, receipt emitter).
- `scripts/run_catalog.py`, `tests/test_weight_doctor.py`,
  `reports/catalog_results.json`, `receipts/`.
- `kaggle/` — self-contained Phase 2 notebooks + committed
  results and verdicts.
