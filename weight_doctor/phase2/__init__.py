"""Phase 2 pivot harness: thinking non-termination in Qwen3.5-0.8B.

Pure logic only — no torch import here, so this package can be
tested locally on CPU. Model execution lives in the Kaggle
notebook, which imports these definitions by copying them
into its self-contained script (Kaggle has no access to this
repo). The Kaggle script and this package must be kept in
sync on prompts and success rules; the validation test in
the build checks the Kaggle script's prompt lists against
this package.
"""
