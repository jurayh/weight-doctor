"""Kill 3 runner: LLM doctor vs scripted baseline on the toy catalog.

    python3 scripts/run_kill3.py --dry-run   # build prompts, measure tokens, no API calls
    python3 scripts/run_kill3.py             # live run (needs estimate-first approval)

The live run calls the Meta Model API (contributor tier by default)
once per catalog case, with at most one retry per case if the
response is not parseable JSON or the diff fails validation. Every
call is checked against the cap BEFORE it is made (see
weight_doctor/meta_client.py); the run stops the moment the cap
would be crossed. Results and receipts go to reports/ and
receipts/kill3/. The verdict against the pre-registered bar in
KILL3_PLAN.md is computed from the saved per-case rows, not from
console output.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weight_doctor.catalog import build_catalog  # noqa: E402
from weight_doctor.doctor import propose_repair  # noqa: E402
from weight_doctor.llm_doctor import (  # noqa: E402
    LLMDoctorBadDiff, build_prompt, parse_diff, run_llm_doctor)
from weight_doctor.meta_client import (  # noqa: E402
    IN_RATE_CONTRIBUTOR, IN_RATE_STANDARD, MAX_OUTPUT_TOKENS,
    OUT_RATE_CONTRIBUTOR, OUT_RATE_STANDARD, MetaClient)
from weight_doctor.policy import apply_diff, diff_stats  # noqa: E402
from weight_doctor.verifier import verify_case  # noqa: E402

MAX_RETRIES = 1
# Output tokens are budgeted at the max-output cap: the diff JSON is
# small, but the estimate must not assume a short reply.
EST_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS


def score_case(data, diff, cause, attempts, spent_usd):
    """Build one case's result row from a parsed diff (or None).

    Uses the verifier's actual return keys (target_after_fail /
    breakage_after_fail are episode lists, not counts). The first
    live run with a valid diff crashed here on invented key names
    and lost the model's reply, because results were only written
    at the end; rows are now saved after every case (see main).
    """
    row = {"attempts": attempts, "cause": cause, "spent_usd": spent_usd}
    if diff is None:
        row.update({"fixed": False, "reason": "no valid diff"})
        return row
    edited = apply_diff(data["broken"], diff)
    v = verify_case(data["broken"], edited, data["target"], data["breakage"])
    stats = diff_stats(data["broken"], edited)
    row.update({
        "fixed": bool(v["fixed"]),
        "weights_changed": stats["changed"],
        "l2": stats["l2_norm"],
        "diff": [[m, r, c, val] for (m, r, c, val) in diff],
        "verifier": {
            "fixed": bool(v["fixed"]),
            "target_after": v["target_total"] - len(v["target_after_fail"]),
            "target_total": v["target_total"],
            "breakage_after": v["breakage_total"] - len(v["breakage_after_fail"]),
            "breakage_total": v["breakage_total"],
        },
    })
    return row


def save_results(results, path="reports/kill3_results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def measure_prompt_tokens(prompt):
    """Token count of the exact prompt text, tiktoken cl100k.

    This is a measurement of the real prompt, not a guess — but it
    is the cl100k tokenizer, not the model provider's own counter.
    The provider's `usage` field is the authoritative count after
    the run, and the cap is enforced against a 1.4x upper bound, so
    a tokenizer mismatch cannot silently blow the budget.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


def price(in_tokens, out_tokens, contributor=True):
    in_rate = IN_RATE_CONTRIBUTOR if contributor else IN_RATE_STANDARD
    out_rate = OUT_RATE_CONTRIBUTOR if contributor else OUT_RATE_STANDARD
    return in_tokens * in_rate + out_tokens * out_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts and measure tokens; make no API calls")
    ap.add_argument("--cap-usd", type=float, default=0.10,
                    help="hard spend cap in USD (default 0.10)")
    ap.add_argument("--model", default="muse-spark-1.3-contributor",
                    help="model id (contributor tier by default)")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="run only these case ids (e.g. --cases dead_right "
                         "for the 1-case smoke test)")
    args = ap.parse_args()

    catalog = build_catalog()
    # The bar is always the FULL catalog's scripted total (41), even
    # for a --cases smoke run on a subset.
    scripted_total = 0
    for case_id, data in catalog.items():
        _, diff = propose_repair(data["broken"])
        scripted_total += len(diff)
    if args.cases:
        unknown = sorted(set(args.cases) - set(catalog))
        if unknown:
            raise SystemExit(f"unknown cases: {unknown}")
        catalog = {cid: catalog[cid] for cid in args.cases}
    print(f"Scripted baseline total weights changed: {scripted_total} "
          f"(bar: LLM must fix 4/4 with strictly fewer)")

    prompts = {cid: build_prompt(data["broken"]) for cid, data in catalog.items()}
    total_in = sum(measure_prompt_tokens(p) for p in prompts.values())
    n_calls = len(prompts) * (1 + MAX_RETRIES)  # worst case: every case retries once
    est_in = total_in * (1 + MAX_RETRIES)  # worst case, all retries fire
    est_out = n_calls * EST_OUTPUT_TOKENS
    print(f"Prompts: {len(prompts)} cases, {total_in} input tokens total "
          f"(cl100k, measured on the exact prompt text)")
    print(f"Worst case ({n_calls} calls): {est_in} in / {est_out} out tokens")
    print(f"  contributor tier: ${price(est_in, est_out, True):.4f}")
    print(f"  standard tier:    ${price(est_in, est_out, False):.4f}")
    print(f"  cap: ${args.cap_usd:.2f}")
    if args.dry_run:
        for cid, p in prompts.items():
            print(f"  {cid}: {measure_prompt_tokens(p)} in tokens, "
                  f"{len(p)} chars")
        return 0

    client = MetaClient(model=args.model, cap_usd=args.cap_usd)
    results = {"model": args.model, "cap_usd": args.cap_usd,
               "scripted_total_weights": scripted_total, "cases": {},
               "complete": False}
    for case_id, data in catalog.items():
        prompt = prompts[case_id]
        attempts = []
        diff = None
        cause = ""
        for attempt in range(1 + MAX_RETRIES):
            try:
                result = client.complete(prompt)
            except Exception as e:
                attempts.append({"attempt": attempt,
                                 "error": f"{type(e).__name__}: {e}",
                                 "usage": getattr(e, "usage", None)})
                break
            try:
                diff, cause = parse_diff(result["text"])
                attempts.append({"attempt": attempt, "ok": True,
                                 "usage": result["usage"],
                                 "raw_text": result["text"]})
                break
            except LLMDoctorBadDiff as e:
                attempts.append({"attempt": attempt, "ok": False,
                                 "error": str(e),
                                 "usage": result["usage"],
                                 "raw_text": result["text"][:500]})
                diff = None
        row = score_case(data, diff, cause, attempts, client.spent_usd)
        results["cases"][case_id] = row
        # Save after every case: a later crash must not lose an
        # already-paid-for reply the way the first valid-diff run did.
        save_results(results)
        print(f"{case_id}: fixed={row.get('fixed')} "
              f"weights={row.get('weights_changed')} spent=${client.spent_usd:.4f}")

    fixed = sum(1 for r in results["cases"].values() if r.get("fixed"))
    total_weights = sum(r.get("weights_changed", 0)
                        for r in results["cases"].values() if r.get("fixed"))
    results["summary"] = {
        "fixed": fixed,
        "total_weights_changed": total_weights,
        "kill3_pass": bool(fixed == 4 and total_weights < scripted_total),
        "spent_usd": client.spent_usd,
        "prompt_tokens": client.prompt_tokens,
        "completion_tokens": client.completion_tokens,
        "calls": client.calls,
    }
    print(f"Kill 3: fixed {fixed}/4, total weights {total_weights} "
          f"(bar < {scripted_total}), pass={results['summary']['kill3_pass']}, "
          f"spent ${client.spent_usd:.4f} over {client.calls} calls "
          f"({client.prompt_tokens} in / {client.completion_tokens} out tokens)")
    results["complete"] = True
    save_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
