# Weight Doctor Phase 2 — loop hunt (Kaggle free GPU)

**Status: case-finding probe only. Phase 2 has not started, and
no weight is edited here.**

## Why this probe exists

The Nov 4 smoke test passed on compute, but its only loop flag
was a false positive for Phase 2 purposes: the banana prompt
instructed repetition ("repeat 50 times"), and the model
produced 200 bananas to the token cap. That is a stop/count
failure on instructed repetition, not a clean unintended loop
bug, and it must not be cited as one.

A Nov 4 search found no documented unintended-loop case for
the smoke model, Qwen2.5-0.5B-Instruct. The documented loop
case is a different model: Qwen3.5-0.8B in thinking mode. Its
model card warns it is prone to repetition loops, and the
"Sparks of Weight-Space Programming" post (MIT CSAIL, Nov 2026)
reports 85 of its first 96 practice math answers running to
the 32,768-token limit, repeating and never reaching an
answer, under model-card sampling (temperature 0.6, top_p
0.95, top_k 20, presence_penalty 1.5). A third-party rebuilt
edited checkpoint exists, but this probe NEVER loads it: the
probe is base-model only, so a later Phase 2 edit cannot be
accused of copying that edit.

## What is in here

- `loop_hunt.ipynb` — the one file to upload to Kaggle. It is
  self-contained: the whole probe script is in one code cell.
- `loop_hunt.py` — the same script, for reading / diffing.
  You do not need to upload this separately.

## What the probe does (and does not do)

Track A — empirical hunt, smoke model, greedy, 512 tokens:
Qwen/Qwen2.5-0.5B-Instruct, 10 open-ended prompts, none asking
to repeat, count, or list anything again, plus 3 short-QA
controls.

Track B — documented case, candidate switch, 1024 tokens:
Qwen/Qwen3.5-0.8B, thinking mode ON, model-card sampling
above (presence_penalty attempted per generation, and
recorded as omitted if the installed transformers rejects
it — on the Nov 4 Kaggle image it is rejected with
ValueError "model_kwargs are not used", and the probe now
retries without it inside generate(), instead of dying;
that deviation is recorded in `used_sampling` and in each
result's `presence_penalty_used` field), 10 fresh GSM-style
word problems written for this probe (answers recorded in the
script for a later battery; the probe itself only detects
loops, it does not score answers), plus 3 short-QA controls
with the same sampling. Track B does NOT use greedy decoding:
Qwen's docs warn greedy can cause endless repetitions in
thinking models, so a greedy loop there would be a decoding
artifact, not a model bug.

Detector v2, same for both tracks: LOOP only if the
generation hits the token cap without EOS AND a line repeats
4+ times, a 20-word window repeats 3+ times, or a 10-word
window repeats 5+ times. A short repeated phrase that stops
cleanly is not a loop. Every prompt is guard-checked for
repeat instructions before running; if a prompt fails the
guard, the script stops. Detector v2 also fixes an off-by-one
in the smoke detector's window enumeration (it needed one
extra repeat to fire); the smoke results are unaffected,
because the banana case repeated 200 times.

Track B also dumps the model structure (layer count, layer
path, layers 15 and 19 MLP down_proj shapes if present).
Dumping is not editing, and it does not copy Astra's edit:
which layers to change is a Phase 2 doctor's job to find.

It does NOT edit any weight, fix any loop, or score math
answers. Cap is 1024 tokens for Track B, not Astra's 32768:
a true loop attractor should already be repeating well
before 1024. If a prompt only starts looping after 1024,
this probe will miss it, and that limit is stated here.

## Clean-case criteria (pre-registered)

- Track B clean case: >= 4/10 math prompts LOOP-flagged,
  0/3 controls LOOP-flagged, AND at least one math prompt
  LOOP-flagged again on a second run with the seed reset.
- Track A clean case: >= 1/10 hunt prompts LOOP-flagged,
  0/3 controls LOOP-flagged, AND that prompt LOOP-flagged
  again on a second (greedy, deterministic) run.
- Decision rule, fixed in advance: Track B pass -> Phase 2
  switches to Qwen3.5-0.8B with the math loop rate as target
  battery. Else Track A pass -> Phase 2 stays on
  Qwen2.5-0.5B-Instruct with that prompt as target case.
  Else neither pass -> Phase 2 does NOT start on a loop
  case. Do not lower the bar after seeing the outputs.

## Run steps (your Kaggle account, same as the smoke test)

1. Create -> New Notebook -> File -> Import Notebook ->
   upload `loop_hunt.ipynb`.
2. Settings: Accelerator -> GPU T4 x2, Internet -> On.
   (Phone verification is already done, so the GPU option
   should be available; if it is greyed out again, see the
   smoke test README troubleshooting.)
3. Run All. First run downloads two models (~1 GB +
   ~1.6 GB), then generation. Typical case should be well
   under 1 GPU-hour; worst case (many prompts hitting their
   caps, plus re-runs) is still comfortably inside a 12-hour
   Kaggle session, but expect tens of minutes, not seconds.
4. Save Version -> Save & Run All, so the outputs survive.
5. Download `loop_hunt_results.json` and
   `loop_hunt_results.md` from `/kaggle/working/` and send
   them back here.

If you get the accelerate ImportError, same fix as before:
new cell at the top with `!pip install -q -U accelerate
transformers`, then Session -> Restart, then Run All again.

## Warnings you can ignore while it runs

- `presence_penalty rejected ...; retrying without it` — this
  is the designed fallback working, not a failure. The run
  continues without presence_penalty and records the omission.
- `causal_conv1d_fn` / `chunk_gated_delta_rule` /
  `fused_recurrent_gated_delta_rule` falling back to a
  reference PyTorch implementation — Qwen3.5's hybrid
  Gated DeltaNet layers normally use those optimized kernels,
  and Kaggle does not have them installed. The reference
  version is correct but much slower, so Track B takes
  longer than a native-kernel run would. Do NOT install
  those packages mid-run: installing them would need a
  session restart and would lose the run in progress.

If you get the presence_penalty ValueError on an older
notebook copy, re-upload the current `loop_hunt.ipynb`:
that error is now handled inside the probe (retry without
presence_penalty, omission recorded), and Track A results
are saved to `loop_hunt_track_a.json` as soon as Track A
finishes, so a Track B failure can no longer lose them.
A Track B technical failure is now reported as "probe
incomplete", not as a clean-case failure.
