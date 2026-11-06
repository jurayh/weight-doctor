"""Receipt emitter: one document per edit.

A receipt states, in this order: what was diagnosed (with the
weight evidence), the diff (exactly what moved, with counts and
norms), the target battery before/after, and the breakage battery
before/after. It also pins the model (weight hashes), the schema
version, and the toy-only scope label. No scale claims leave the
toy room: every receipt says it is a toy result.
"""

import json

from .policy import SCHEMA_VERSION, weights_hash

VERSION = "0.1.0"
SCOPE_LABEL = ("TOY ONLY: 3x3 grid MLP, 20,612 parameters. "
               "Not a claim about any larger model.")


def _ep(ep):
    return {"agent": ep[0], "goal": ep[1], "hazard": ep[2]}


def build_receipt(case_id, title, ground_truth, diagnoses, diff,
                  verification, broken, edited):
    stats = verification["minimality"]
    receipt = {
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE_LABEL,
        "case_id": case_id,
        "case_title": title,
        "ground_truth_cause": ground_truth,
        "diagnoses": [
            {"fault": d["fault"], "cause": d["cause"],
             "evidence": d["evidence"], "diff_size": len(d["diff"])}
            for d in diagnoses
        ],
        "diff": {
            "entries": [
                {"matrix": name, "row": row, "col": col,
                 "before": before, "after": after}
                for name, row, col, before, after in stats["entries"]
            ],
            "changed": stats["changed"],
            "total_params": stats["total_params"],
            "fraction_changed": stats["fraction_changed"],
            "l2_norm": stats["l2_norm"],
            "linf": stats["linf"],
        },
        "target_battery": {
            "total": verification["target_total"],
            "passing_before_edit": verification["target_before_pass"],
            "failing_after_edit": [_ep(e)
                                    for e in verification["target_after_fail"]],
        },
        "breakage_battery": {
            "total": verification["breakage_total"],
            "failing_after_edit": [_ep(e) for e in
                                    verification["breakage_after_fail"]],
        },
        "minimality_within_bars": verification["minimality_within_bars"],
        "fixed": verification["fixed"],
        "model_hashes": {
            "broken_sha256": weights_hash(broken),
            "edited_sha256": weights_hash(edited),
        },
    }
    return receipt


def receipt_markdown(receipt):
    d = receipt["diff"]
    lines = []
    lines.append(f"# Receipt: {receipt['case_id']} — "
                 f"{receipt['case_title']}")
    lines.append("")
    lines.append(f"> {receipt['scope']}")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    for diag in receipt["diagnoses"]:
        lines.append(f"- **{diag['fault']}**: {diag['cause']}")
    lines.append("")
    lines.append(f"Ground truth (catalog): {receipt['ground_truth_cause']}")
    lines.append("")
    lines.append("## Diff")
    lines.append("")
    lines.append(f"- Changed: {d['changed']} / {d['total_params']} "
                 f"weights ({d['fraction_changed']:.4%})")
    lines.append(f"- Edit L2 norm: {d['l2_norm']:.3f}, "
                 f"Linf: {d['linf']:.3f}")
    lines.append(f"- Broken hash: `{receipt['model_hashes']['broken_sha256'][:16]}…`")
    lines.append(f"- Edited hash: `{receipt['model_hashes']['edited_sha256'][:16]}…`")
    lines.append("")
    lines.append("## Target battery (the failing behaviour)")
    lines.append("")
    t = receipt["target_battery"]
    lines.append(f"- Total: {t['total']}, passing before edit: "
                 f"{t['passing_before_edit']}, failing after edit: "
                 f"{len(t['failing_after_edit'])}")
    lines.append("")
    lines.append("## Breakage battery (everything else)")
    lines.append("")
    b = receipt["breakage_battery"]
    lines.append(f"- Total: {b['total']}, failing after edit: "
                 f"{len(b['failing_after_edit'])}")
    lines.append("")
    lines.append(f"## Verdict: {'FIXED' if receipt['fixed'] else 'NOT FIXED'}")
    lines.append("")
    return "\n".join(lines)


def write_receipt(receipt, json_path, md_path):
    with open(json_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    with open(md_path, "w") as f:
        f.write(receipt_markdown(receipt))
