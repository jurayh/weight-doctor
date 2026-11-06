# Weight Doctor Phase 2 — Kaggle free-GPU smoke test

**Status: smoke test only. Phase 2 has not started.**

This folder is the Kaggle-first compute path, chosen Nov 4, 2026
after the RunPod RTX 4090 / $10 cap proposal was not approved.
It costs $0 in rental: Kaggle's free tier (about 30 GPU hours a
week, T4 hardware, sessions up to 12 hours). The smoke test
itself should use well under 1 GPU-hour.

## What is in here

- `phase2_smoke.ipynb` — the one file to upload to Kaggle. It is
  self-contained: the whole smoke script is in one code cell.
- `phase2_smoke.py` — the same script, for reading / diffing.
  You do not need to upload this separately.

## What the smoke test does (and does not do)

Model, pinned: `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters,
safetensors, loaded in float16 — T4s predate bf16).

1. Confirms a GPU is present and prints its name.
2. Runs 5 loop-prone prompts with greedy decoding, max 200 new
   tokens each, and reports per prompt: tokens generated, whether
   it hit the token cap, and whether the loop detector fired.
3. Applies one tiny weight diff (zero one row of
   `layers[0].self_attn.o_proj.weight`), re-runs one prompt, and
   reports the diff size. This proves edit-as-diff works in
   torch on this GPU. It is not a fix.
4. Writes `phase2_smoke_results.json` + `.md` to the working
   directory.

It does **not** diagnose a loop bug, fix anything, or claim
Phase 2 passes. A NO_LOOP on all 5 prompts is still a passing
smoke test: it means the model did not loop on those prompts,
not that loops do not exist. The loop rule is in the script
docstring (token cap + repeated 20-word window, or a line
repeated 4+ times).

## Run steps (your Kaggle account)

I cannot run this from my machine: there is no Kaggle connector
here, and I have no Kaggle account or API token for you (none is
in memory). The run happens in your browser, under your account.
It should take a few minutes of your time plus the run itself.

1. Kaggle → Settings → Phone Verification, once. Internet in
   notebooks is gated behind this; without it the model
   download fails.
2. Create → New Notebook → File → Import Notebook → upload
   `phase2_smoke.ipynb`.
3. Right panel / Settings: Accelerator → GPU T4 x2, Internet →
   On. (One GPU is used; quota is charged per hour, not per GPU.)
4. Run All. First run downloads ~1 GB of model weights, then
   generation takes a few minutes.
5. Save Version → Save & Run All, so the outputs survive the
   session. Interactive-session files do not survive on their own.
6. Download `phase2_smoke_results.json` and
   `phase2_smoke_results.md` from `/kaggle/working/` (or from the
   saved version's output) and send them back here.

## If GPU T4 x2 is greyed out

Check in this order; the first one is the usual cause:

1. Phone verification. Kaggle gates accelerators behind a
   verified phone number, and the GPU option stays unavailable
   until you verify. Go to Kaggle → Settings → Phone
   Verification and confirm it says Verified, then reload the
   notebook. (This step is also required for Internet On.)
2. Session already running. Stop / end the current session
   first, then change the accelerator, then start again —
   the dropdown is locked while a session is active.
3. Weekly quota. Check that you have GPU hours left for the
   week (about 30 hours a week on the free tier); an exhausted
   quota also greys the option out.
4. Browser. If you are on Brave or a browser blocking the
   selector, try Chrome.
5. Do not create a second Kaggle account to get more quota:
   Kaggle's guidelines allow immediate bans for duplicate
   accounts.

T4 x2 is now the only GPU Kaggle offers in notebooks (P100
was retired Sep 15, 2026), so there is no other GPU type to
switch to inside Kaggle. If verification is done, the session
is stopped, and quota remains, but the option is still
greyed out, that is a failed smoke test on Kaggle — report
exactly what Settings shows and fall back to RunPod A5000 /
$5 cap.

## If you get the accelerate ImportError

Error: `ImportError: cannot import name 'dispatch_model' from
partially initialized module 'accelerate.big_modeling'`.

This is a mismatched accelerate/transformers pair in the Kaggle
image, not this script. Fix, in order:

1. New cell at the top of the notebook:
   `!pip install -q -U accelerate transformers`
2. Restart the session (Session → Restart). This step is
   required: installing in the same session does not fix a
   partially initialized module.
3. Run All again.

If it still fails, send back the output of
`!pip show accelerate transformers torch` so exact versions
can be pinned. Also check there is no file named
`accelerate.py` in `/kaggle/working/` shadowing the package.

## If the download fails

Fallback, in order, and say which one you used:

1. Check Internet is On and phone verification is done, then
   re-run. This is the most common failure.
2. Add the model as an input instead of downloading it
   in-notebook (Add Input → model/dataset), set `MODEL_ID` in
   the first code cell to that local path, and re-run.
3. Do not substitute a different model size without saying so:
   the smoke test is pinned to Qwen2.5-0.5B-Instruct, and a
   different model changes what the result means.

## Success bar (pre-registered for this smoke test)

- GPU present and named in the results.
- All 5 baseline prompts complete with token counts.
- Diff check completes: 1 row changed, generation still runs.
- Both result files are returned.

Anything else — no GPU, download failure, import missing — is
a failed smoke test, and Phase 2 does not start on Kaggle on
the back of it. The RunPod A5000 / $5 cap and Vast.ai spot
options from Nov 4 remain the fallbacks, and the VM CPU test
remains paused, not dropped.
