"""Receipt emitter for Phase 2 pivot candidates.

One JSON + one Markdown receipt per candidate, mirroring
Phase 1's receipts/ format: the candidate's diff stats,
per-prompt target and breakage results, and the Kill P1/P2
verdict. A failed candidate gets a receipt too — a failed
candidate is a result, not a deleted row.
"""

import json


def build_receipt(candidate_name, family, diff_stats,
                  target_results, breakage_results, verdict,
                  environment, sampling):
    """target_results / breakage_results: lists of dicts with
    at least prompt, success, closed, correct, new_tokens."""
    return {
        "candidate": candidate_name,
        "family": family,
        "scope": ("Qwen3.5-0.8B, thinking mode ON, Kaggle T4, "
                  "1024-token cap, this battery only. No claim "
                  "about other models, modes, or batteries."),
        "diff": diff_stats,
        "target": {
            "hits": verdict["target_hits"],
            "total": verdict["target_total"],
            "per_prompt": target_results,
        },
        "breakage": {
            "regressions": verdict["breakage_regressions"],
            "regression_count": verdict["breakage_regression_count"],
            "per_prompt": breakage_results,
        },
        "verdict": {
            "kill_p1_pass": verdict["kill_p1_pass"],
            "kill_p2_pass": verdict["kill_p2_pass"],
            "passes_both": verdict["passes_both"],
        },
        "environment": environment,
        "sampling": sampling,
    }


def receipt_markdown(receipt):
    lines = []
    lines.append(f"# Receipt — {receipt['candidate']}\n")
    lines.append(f"Family: {receipt['family']}\n")
    lines.append(f"Scope: {receipt['scope']}\n")
    d = receipt["diff"]
    lines.append(f"Diff: {d.get('weights_changed')} weights changed "
                 f"({d.get('weight_fraction')}), "
                 f"L2={d.get('l2_norm')}, "
                 f"layer-limited={d.get('layer_limited')}\n")
    t = receipt["target"]
    lines.append(f"Target: {t['hits']}/{t['total']}\n")
    for r in t["per_prompt"]:
        lines.append(f"- {'PASS' if r['success'] else 'fail'}: "
                     f"{r['prompt'][:70]} "
                     f"(closed={r['closed']}, correct={r['correct']}, "
                     f"tokens={r['new_tokens']})\n")
    b = receipt["breakage"]
    lines.append(f"\nBreakage regressions: {b['regression_count']} "
                 f"at indices {b['regressions']}\n")
    v = receipt["verdict"]
    lines.append(f"Kill P1: {v['kill_p1_pass']}, "
                 f"Kill P2: {v['kill_p2_pass']}\n")
    return "".join(lines)


def write_receipt(receipt, json_path, md_path):
    with open(json_path, "w") as f:
        json.dump(receipt, f, indent=2)
    with open(md_path, "w") as f:
        f.write(receipt_markdown(receipt))
