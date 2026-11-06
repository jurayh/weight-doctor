"""Run the whole toy catalog end to end and emit receipts.

Usage: python3 scripts/run_catalog.py
Writes reports/catalog_results.json and one receipt pair
(receipts/<case>.json, receipts/<case>.md) per case, and prints a
summary table. Pure local CPU, no network, no model API calls.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weight_doctor.catalog import build_catalog
from weight_doctor.doctor import propose_repair
from weight_doctor.policy import apply_diff, weights_hash
from weight_doctor.receipt import build_receipt, write_receipt
from weight_doctor.verifier import verify_case


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    receipts_dir = os.path.join(root, "receipts")
    reports_dir = os.path.join(root, "reports")
    os.makedirs(receipts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    catalog = build_catalog()
    summary = {}
    print(f"{'case':<14} {'target':>10} {'breakage':>10} "
          f"{'changed':>8} {'frac':>8} {'l2':>7}  verdict")
    for case_id, data in catalog.items():
        broken = data["broken"]
        diagnoses, diff = propose_repair(broken)
        edited = apply_diff(broken, diff)
        verification = verify_case(broken, edited,
                                   data["target"], data["breakage"])
        # Diagnosis must match ground truth: the catalog's planted
        # fault id is the diagnosis fault id for every case here.
        receipt = build_receipt(
            case_id, data["case"]["title"],
            data["case"]["ground_truth_cause"],
            diagnoses, diff, verification, broken, edited)
        write_receipt(receipt,
                      os.path.join(receipts_dir, f"{case_id}.json"),
                      os.path.join(receipts_dir, f"{case_id}.md"))
        stats = verification["minimality"]
        print(f"{case_id:<14} "
              f"{verification['target_total'] - len(verification['target_after_fail'])}/{verification['target_total']:>7} "
              f"{verification['breakage_total'] - len(verification['breakage_after_fail'])}/{verification['breakage_total']:>7} "
              f"{stats['changed']:>8} {stats['fraction_changed']:>8.4%} "
              f"{stats['l2_norm']:>7.2f}  "
              f"{'FIXED' if verification['fixed'] else 'NOT FIXED'}")
        summary[case_id] = {
            "title": data["case"]["title"],
            "diagnoses": [d["fault"] for d in diagnoses],
            "healthy_pass": data["healthy_pass"],
            "broken_pass": data["broken_pass"],
            "episodes_total": data["episodes_total"],
            "target_total": verification["target_total"],
            "target_after_fail": len(verification["target_after_fail"]),
            "breakage_total": verification["breakage_total"],
            "breakage_after_fail": len(verification["breakage_after_fail"]),
            "changed": stats["changed"],
            "total_params": stats["total_params"],
            "fraction_changed": stats["fraction_changed"],
            "l2_norm": stats["l2_norm"],
            "fixed": verification["fixed"],
            "edited_sha256": weights_hash(edited),
        }
    fixed = sum(1 for v in summary.values() if v["fixed"])
    summary["_overall"] = {
        "cases": len(catalog),
        "fixed": fixed,
        "kill1_bar_fixed_at_least": 3,
        "kill1_pass": fixed >= 3,
    }
    with open(os.path.join(reports_dir, "catalog_results.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nFixed {fixed}/{len(catalog)} cases "
          f"(kill 1 bar: at least 3, with zero breakage regressions).")


if __name__ == "__main__":
    main()
