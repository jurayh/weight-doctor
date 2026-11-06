# Weight Doctor Phase 2 — Stage 1 (baseline + diagnosis)

Model: Qwen/Qwen3.5-0.8B on Tesla T4

Baseline reproduced (0/9 target, 4/4 known breakage): True

Baseline target: 0/9
Baseline known breakage: 4/4
Baseline new breakage: 4/10

Diagnosis: marker rates, </think> trajectory, thinking-OFF, and layer ablation are in the JSON.

No weight was edited in Stage 1. Ablation used forward hooks only; hooks were removed after each layer, and model weights were never changed.
