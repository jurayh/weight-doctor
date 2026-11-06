# Weight Doctor — cap-sensitivity probe (Kaggle free GPU)

**Status: gate probe only. Phase 2 has not started, and no
weight is edited here.**

## Why this probe exists

Loop hunt Track B (Nov 4, 2026): Qwen3.5-0.8B in thinking
mode closed thinking on only 4/13 prompts at a 1024-token
cap. The other 9 hit the cap still thinking, with the right
answer already in the thinking text. That could be genuine
non-termination, or it could be that this model thinks
verbosely and would close at 1500 or 3000 tokens. This
probe changes exactly one variable — the cap, to 4096 —
to tell those apart before any edit is attempted.

## What is in here

- `cap_sensitivity.ipynb` — the one file to upload to
  Kaggle. It is self-contained: the whole probe script is
  in one code cell.
- `cap_sensitivity.py` — the same script, for reading /
  diffing. You do not need to upload this separately.

## What the probe does (and does not do)

Model, pinned: `Qwen/Qwen3.5-0.8B`, thinking mode ON,
same sampling as loop-hunt Track B actually used
(temperature 0.6, top_p 0.95, top_k 20; presence_penalty
attempted with the same per-call fallback, omission
recorded), same seed (0), so the first 1024 tokens should
reproduce Track B's run.

Prompts, verbatim from Track B (verified against
`loop_hunt_results.json`, not copied by eye):
- 9 non-terminating prompts: the 7 math problems and 2
  trivial controls that hit the 1024 cap without closing.
- 4 sanity prompts: the 3 math problems and 1 control
  that already closed (666, 350, 732, 134 tokens). They
  should close at the same counts again, proving the
  longer cap did not change behaviour on prompts that
  already terminated.

Closure = `</think>` appears in the output AND generation
stopped before the 4096 cap. A prompt that closes gets a
second seeded run to confirm closure; a prompt that hits
the 4096 cap is not re-run (it would cost the most time
and add nothing to the gate).

Results are written after every prompt
(`cap_sensitivity_results.json`), so a crash or session
end cannot lose completed prompts. Versions of torch,
transformers, and accelerate are logged at the start.

It does NOT edit any weight, and it does NOT judge
answer accuracy as the gate — accuracy after `</think>`
is recorded for closed prompts only, as breakage-battery
material for later.

## Pre-registered gate

- **PIVOT:** >= 5 of the 9 non-terminating prompts are
  still unclosed at 4096 tokens. Non-termination is real;
  Phase 2 may pivot to it, with close-rate at 1024 as the
  target battery and answer accuracy + short-QA as
  breakage.
- **NO PIVOT:** <= 4 of the 9 are still unclosed at 4096.
  The 1024-cap reading was mostly a cap artifact; the
  pivot dies and the loop line stays closed. A 3–4
  unclosed result is still NO PIVOT under this rule.
- **INCOMPLETE:** any already-closed sanity prompt fails
  to close at 4096. Environment changed; re-run before
  judging anything.

## Run steps (your Kaggle account, same as before)

1. Create -> New Notebook -> File -> Import Notebook ->
   upload `cap_sensitivity.ipynb`.
2. Settings: Accelerator -> GPU T4 x2, Internet -> On.
3. Run All. First run downloads the ~1.6 GB model, then
   generation. Worst case (all 9 prompts hitting 4096
   tokens) is about 35–40 minutes of generation at the
   loop hunt's ~20 tokens/s on reference kernels, plus
   download time; typical case is less, since prompts
   that close stop early. Still well inside a 12-hour
   Kaggle session.
4. Save Version -> Save & Run All, so the outputs survive.
5. Download `cap_sensitivity_results.json` and
   `cap_sensitivity_results.md` from `/kaggle/working/`
   and send them back here.

## Do not change the environment for this probe

Do NOT pip-upgrade transformers before this run. A
version change would confound the cap effect with an
environment effect, which is exactly what this probe is
trying to isolate. The one exception is the known
accelerate ImportError fix (new cell with
`!pip install -q -U accelerate transformers`, then
Session -> Restart, then Run All) — if that fix is
needed, the version change is logged in the results and
must be noted when the gate is judged.
