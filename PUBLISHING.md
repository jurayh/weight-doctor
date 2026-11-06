# Publishing checklist

This file is for the maintainer (Yuriy), not for readers. It
records what was checked before the repo went public, and the
steps to push it.

## Pre-push checks (done Nov 5, 2026)

- [x] License: MIT, copyright "Weight Doctor contributors"
      (`LICENSE`). No personal name anywhere in the repo.
- [x] Secrets scan: no API keys, tokens, passwords, emails, or
      personal names in any tracked file (checked Nov 5).
      The only "token" hits are tokenizer/token-cap words.
- [x] No `__pycache__` / `*.pyc` tracked (`.gitignore` covers
      them; `git ls-files | grep pyc` returns nothing).
- [x] No files over 500 KB; repo is ~2.5 MB total.
- [x] Tests pass: `python3 tests/test_weight_doctor.py` → 6/6.
      The test file now has a stdlib runner, so it prints
      results when run directly (it previously defined tests
      but did nothing on direct execution).
- [x] README rewritten for publishing: Phase 1 + Phase 2,
      results table, generalisation probe, honest limits,
      C6 forcing-closure caveat, tied-embedding disclosure.
      All README links checked; none are broken.
- [x] Results committed with receipts:
      `kaggle/phase2-pivot/results-rerun/` (Stage 2, all 6
      candidates) and `kaggle/phase2-pivot/gen-probe-results/`
      (generalisation probe). First-run and crashed-rerun
      console-only results are reported in
      `PHASE2_PIVOT_RESULTS.md` with that caveat attached;
      their receipts were lost to Kaggle's per-session
      working directory and are not claimed to exist.

## Open items (decisions, not blockers)

- Repo name: `weight-doctor` is the working name (folder and
  LICENSE holder). Create it as `jurayh/weight-doctor` unless
  you want a different name.
- The GitHub token cannot create repos (403, recorded Oct 1),
  so repo creation is your web click. Pushing to the repo
  once it exists works via the usual push helper.
- GitHub Pages / repo topics are also manual web-UI steps for
  this token (same 403 class); not needed for publishing.

## Push steps

1. Create the repo on GitHub (public, empty — no README,
   no .gitignore, no license from the web UI, all three
   already exist here).
2. From this directory, push the local tree to the new remote
   (use the push helper with an explicit commit message;
   the tree is already committed and clean — verify with
   `git status` and `git log --oneline -3` first).
3. Verify on GitHub: README renders, `LICENSE` shows MIT,
   and the receipt JSONs open under
   `kaggle/phase2-pivot/results-rerun/phase2_stage2_receipts/`.
