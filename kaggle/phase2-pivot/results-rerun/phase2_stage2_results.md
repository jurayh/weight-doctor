# Weight Doctor Phase 2 — Stage 2 (candidate edits)

Model: Qwen/Qwen3.5-0.8B on Tesla T4

Baseline: target 0/9, known breakage 4/4, new breakage 4/10
Token resolution: resolved </think> id = 248069 (convert=248069, encode=[248069])

Candidates evaluated: 6, overall pass (P1+P2): True

- C1_subgroup_ablation: target 5/9, regressions 2, P1=False P2=True
- C2_group_ablation: target 6/9, regressions 3, P1=False P2=True
- C_direction_0.25: target 3/9, regressions 3, P1=False P2=True
- C_direction_0.5: target 5/9, regressions 1, P1=False P2=True
- C_direction_1.0: target 3/9, regressions 2, P1=False P2=True
- C6_lm_head_bias: target 5/9, regressions 0, P1=True P2=True

Not evaluated:

Receipts in phase2_stage2_receipts/ (JSON + MD each).
No claim is made beyond this model, mode, sampling, and battery (Kill P4).
