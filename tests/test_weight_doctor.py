"""Tests for the toy harness. No network, no model APIs, local CPU."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weight_doctor.catalog import build_catalog
from weight_doctor.doctor import propose_repair
from weight_doctor.game import all_episodes, run_episode
from weight_doctor.policy import (apply_diff, diff_stats, healthy_policy,
                                 make_policy_fn, total_params,
                                 weights_hash)
from weight_doctor.verifier import verify_case


def test_enumeration_is_exact():
    eps = all_episodes()
    assert len(eps) == 504
    assert len(set(eps)) == 504


def test_healthy_passes_everything():
    fn = make_policy_fn(healthy_policy())
    failed = [ep for ep in all_episodes() if not run_episode(fn, ep)]
    assert failed == [], f"healthy policy fails {failed[:5]}"


def test_total_params():
    assert total_params(healthy_policy()) == 20612


def test_catalog_targets_nonempty():
    catalog = build_catalog()
    assert set(catalog) == {"blind_goal", "blind_hazard",
                            "dead_right", "stuck_center"}
    for case_id, data in catalog.items():
        assert data["target"], f"{case_id} has empty target battery"
        assert data["healthy_pass"] == 504


def test_doctor_fixes_all_cases_surgically():
    catalog = build_catalog()
    for case_id, data in catalog.items():
        diagnoses, diff = propose_repair(data["broken"])
        assert diagnoses, f"{case_id}: doctor proposed nothing"
        edited = apply_diff(data["broken"], diff)
        # Doctor must not rewrite the model in place.
        assert weights_hash(data["broken"]) != weights_hash(edited)
        v = verify_case(data["broken"], edited,
                        data["target"], data["breakage"])
        assert v["fixed"], f"{case_id} not fixed: {v}"
        stats = v["minimality"]
        assert stats["changed"] <= 20, case_id
        assert stats["fraction_changed"] <= 0.02, case_id


def test_diff_is_not_a_rewrite():
    catalog = build_catalog()
    data = catalog["dead_right"]
    _, diff = propose_repair(data["broken"])
    edited = apply_diff(data["broken"], diff)
    stats = diff_stats(data["broken"], edited)
    assert stats["changed"] == len(diff)
    # Applying the diff twice is idempotent: same edited weights.
    assert weights_hash(apply_diff(data["broken"], diff)) == \
        weights_hash(edited)


if __name__ == "__main__":
    # Plain-stdlib runner, so the suite runs without pytest:
    #   python3 tests/test_weight_doctor.py
    # (pytest also discovers these test_* functions.)
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
