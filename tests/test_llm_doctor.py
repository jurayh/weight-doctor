"""Tests for the kill-3 LLM-doctor harness. No network, no model APIs.

A stub client returns the scripted doctor's own diff as JSON, which
proves the prompt -> parse -> validate -> apply -> verify pipeline
end to end without spending anything. It does NOT prove the real
LLM can do the same; that is what the approved paid run measures.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weight_doctor.catalog import build_catalog  # noqa: E402
from weight_doctor.doctor import propose_repair  # noqa: E402
from weight_doctor.llm_doctor import (  # noqa: E402
    LLMDoctorBadDiff, build_evidence, build_prompt, parse_diff,
    run_llm_doctor)
from weight_doctor.policy import apply_diff  # noqa: E402
from weight_doctor.verifier import verify_case  # noqa: E402


class StubClient:
    """Returns a fixed JSON diff, shaped like a model reply."""

    def __init__(self, diff, cause="stub"):
        self.diff = diff
        self.cause = cause
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        raw = [[m, r, c, v] for (m, r, c, v) in self.diff]
        return {"text": json.dumps({"cause": self.cause, "diff": raw}),
                "usage": {"input_tokens": 0, "output_tokens": 0}}


def test_evidence_matches_scripted_doctor_inputs():
    catalog = build_catalog()
    ev = build_evidence(catalog["blind_goal"]["broken"])
    assert ev["w1_group_norms"]["goal"] == 0.0
    assert ev["w1_group_norms"]["agent"] > 0
    assert ev["w1_group_norms"]["hazard"] > 0
    ev = build_evidence(catalog["dead_right"]["broken"])
    right = [r for r in ev["wout_rows"] if r["action"] == "RIGHT"][0]
    assert right["row_norm"] == 0.0 and right["bias"] == -5.0


def test_prompt_contains_evidence_and_schema():
    catalog = build_catalog()
    prompt = build_prompt(catalog["blind_goal"]["broken"])
    assert '"goal":0.0' in prompt  # compact separators in the evidence JSON
    assert "Healthy Wout" in prompt
    assert "Return ONLY JSON" in prompt
    # The null-col convention for bias vectors must be stated in the
    # prompt: the first live smoke test failed on exactly this, with
    # the model writing ["bout", 0, 0, 0.0] because it was never told.
    assert '["bout", 0, null, 0.0]' in prompt
    # The Wout wiring rule must be stated too: a later smoke run had
    # the model guess a contiguous block (cols 0-8) for the RIGHT row
    # because the prompt never gave the sparse schema pattern.
    assert "Wout[d, c*4+d] = 2.0" in prompt
    assert "Wout[d, 36+c*4+d] = 1.0" in prompt


def test_parse_diff_accepts_null_col_for_bias():
    diff, _ = parse_diff(json.dumps(
        {"cause": "x", "diff": [["bout", 0, None, 0.0]]}))
    assert diff == [("bout", 0, None, 0.0)]


def test_parse_diff_rejects_bad_entries():
    for bad in [
        "not json",
        json.dumps({"cause": "x"}),
        json.dumps({"diff": [["W1", 0, 0, 1.0], ["W1", 0, 0, 2.0]]}),
        json.dumps({"diff": [["W9", 0, 0, 1.0]]}),
        json.dumps({"diff": [["W1", 999, 0, 1.0]]}),
        json.dumps({"diff": [["bout", 0, 0, 1.0]]}),
        json.dumps({"diff": [["W1", 0, 0, float("inf")]]}),
    ]:
        try:
            parse_diff(bad)
        except LLMDoctorBadDiff:
            continue
        raise AssertionError(f"bad diff accepted: {bad[:80]}")


def test_stub_client_pipeline_fixes_all_cases():
    catalog = build_catalog()
    for case_id, data in catalog.items():
        _, diff = propose_repair(data["broken"])
        client = StubClient(diff)
        got, cause, raw, usage = run_llm_doctor(data["broken"], client=client)
        assert got == diff, case_id
        edited = apply_diff(data["broken"], got)
        v = verify_case(data["broken"], edited, data["target"], data["breakage"])
        assert v["fixed"], case_id


def test_score_case_row_uses_real_verifier_keys():
    # Regression test: the first live run with a valid diff crashed
    # in the runner on invented verifier key names (target_after)
    # and lost the model's reply, because results were only written
    # at the end. score_case must use the verifier's real keys.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_kill3", os.path.join(os.path.dirname(__file__), "..",
                                  "scripts", "run_kill3.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    catalog = build_catalog()
    data = catalog["dead_right"]
    _, diff = propose_repair(data["broken"])
    row = mod.score_case(data, diff, "stub", [], 0.0)
    assert row["fixed"] is True
    assert row["weights_changed"] == 13
    assert row["verifier"]["target_after"] == 195
    assert row["verifier"]["target_total"] == 195
    assert row["verifier"]["breakage_after"] == 309
    assert row["verifier"]["breakage_total"] == 309
    row = mod.score_case(data, None, "", [], 0.0)
    assert row["fixed"] is False and row["reason"] == "no valid diff"


def test_run_without_client_refuses():
    catalog = build_catalog()
    try:
        run_llm_doctor(catalog["dead_right"]["broken"], client=None)
    except RuntimeError as e:
        assert "not run" in str(e)
    else:
        raise AssertionError("run without client did not refuse")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"PASS {name}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
