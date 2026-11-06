"""Verifier for Phase 2 pivot candidates (Kills P1–P2).

Inputs are per-prompt success booleans, as produced by
batteries.check_success. This module never sees weights
or model outputs, so it can be tested locally.

Kill P1: at least 5/9 target successes AND zero breakage
regressions, where a regression = a breakage prompt that
passed at baseline and fails after the edit. Breakage
prompts that already failed at baseline cannot regress;
they are reported, not counted.
Kill P2: fraction of weights changed <= 0.001 (0.1%) AND
the edit is layer/column/head-limited (recorded as a
boolean flag by the candidate runner; this module checks
the flag, it does not verify the edit's shape itself).
L2 is reported, with no absolute bar (see the plan).
Kill P3 (LLM doctor) and P4 (no scale claims) are not
code-checked: P3 is untested by design in this build,
and P4 is a labelling rule for documents.
"""

TARGET_BAR = 5
TARGET_TOTAL = 9
WEIGHT_FRACTION_BAR = 0.001


def evaluate_candidate(target_success, baseline_breakage,
                       candidate_breakage, weights_changed,
                       total_weights, layer_limited):
    """target_success: list of 9 bools.
    baseline_breakage / candidate_breakage: lists of equal
    length (14) of bools. Returns a verdict dict."""
    if len(target_success) != TARGET_TOTAL:
        raise ValueError("target battery must have 9 prompts")
    if len(baseline_breakage) != len(candidate_breakage):
        raise ValueError("breakage lists must match in length")
    target_hits = sum(1 for s in target_success if s)
    regressions = [i for i, (b, c) in
                   enumerate(zip(baseline_breakage, candidate_breakage))
                   if b and not c]
    weight_fraction = (weights_changed / total_weights
                       if total_weights else None)
    p1 = target_hits >= TARGET_BAR and len(regressions) == 0
    p2 = (weight_fraction is not None
          and weight_fraction <= WEIGHT_FRACTION_BAR
          and bool(layer_limited))
    return {
        "target_hits": target_hits,
        "target_total": TARGET_TOTAL,
        "breakage_regressions": regressions,
        "breakage_regression_count": len(regressions),
        "weight_fraction": weight_fraction,
        "weights_changed": weights_changed,
        "layer_limited": bool(layer_limited),
        "kill_p1_pass": p1,
        "kill_p2_pass": p2,
        "passes_both": p1 and p2,
    }
