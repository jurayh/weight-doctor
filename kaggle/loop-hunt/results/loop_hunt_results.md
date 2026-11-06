# Weight Doctor Phase 2 — loop hunt

GPU: Tesla T4

Track A (Qwen2.5-0.5B-Instruct, greedy): 0/10 loops, controls 0/3, clean case: False

Track B (Qwen3.5-0.8B, thinking, model-card sampling): 1/10 loops, controls 2/3, clean case: False

Decision: No clean case: Phase 2 does NOT start on a loop case. The negative result stands; do not lower the bar.

This is a case-finding probe, not Phase 2. No loop was claimed fixed, and no weight was edited.
