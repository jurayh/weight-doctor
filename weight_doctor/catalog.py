"""The failure catalog: planted failures with ground truth.

Each case is one weight edit applied to the healthy policy, with a
recorded ground-truth cause. The target and breakage batteries are
not hand-picked: they are computed by running the healthy and the
broken policy over all 504 episodes.

  target battery   = episodes the healthy policy passes and the
                     broken policy fails (the failing behaviour)
  breakage battery = episodes the healthy policy passes and the
                     broken policy also passes (everything else
                     that must stay green after the fix)

A case is only in the catalog if its target battery is non-empty:
a planted failure that changes nothing is not a failure.
"""

import numpy as np

from .game import NCELL, all_episodes, run_episode
from .policy import (GOAL_PRESENT_UNIT, copy_weights, healthy_policy,
                     make_policy_fn)

CASES = {}


def _case(case_id, title, cause, plant_fn):
    CASES[case_id] = {"id": case_id, "title": title,
                      "ground_truth_cause": cause, "plant": plant_fn}


def _plant_blind_goal(w):
    w = copy_weights(w)
    w["W1"][:, NCELL:2 * NCELL] = 0.0  # all goal input columns
    return w


def _plant_blind_hazard(w):
    w = copy_weights(w)
    w["W1"][:, 2 * NCELL:3 * NCELL] = 0.0  # all hazard input columns
    return w


def _plant_dead_right(w):
    w = copy_weights(w)
    w["Wout"][0, :] = 0.0  # RIGHT output row
    w["bout"][0] = -5.0
    return w


def _plant_stuck_center(w):
    # Spurious strong weight: at the centre cell, the fallback UP
    # unit drives UP with weight 11 instead of the schema value 1,
    # so the policy always leaves the centre upward and, when the
    # goal lies south, walks straight back into the centre: a loop,
    # i.e. a local minimum.
    w = copy_weights(w)
    centre = 4  # cell (1, 1)
    fb_up_centre = 36 + centre * 4 + 3  # direction 3 = UP
    w["Wout"][3, fb_up_centre] = 11.0
    return w


_case("blind_goal",
      "Policy ignores the goal channel",
      "All goal input columns in W1 are zero: the goal pass-through "
      "units and goal_present never fire, so neither the reducing "
      "moves nor the fallback fire.",
      _plant_blind_goal)
_case("blind_hazard",
      "Policy ignores the hazard channel",
      "All hazard input columns in W1 are zero: the hazard "
      "pass-through units never fire, so the hazard term in the "
      "safe_reducing units is missing and the policy steps onto "
      "hazards on its greedy path.",
      _plant_blind_hazard)
_case("dead_right",
      "Policy has a dead RIGHT action",
      "The RIGHT output row is zeroed and its bias is -5, so RIGHT "
      "can never win argmax; any goal east of the start is "
      "unreachable.",
      _plant_dead_right)
_case("stuck_center",
      "Policy is stuck in a local minimum at the centre",
      "One spurious Wout weight (centre fallback-UP, 1 -> 11) makes "
      "UP dominate at the centre whatever the goal is, so paths "
      "through the centre with a southern goal oscillate "
      "centre -> north -> centre and loop.",
      _plant_stuck_center)


def build_catalog():
    """Run healthy and broken policies; return per-case batteries."""
    healthy = healthy_policy()
    healthy_fn = make_policy_fn(healthy)
    episodes = all_episodes()
    healthy_pass = set()
    for ep in episodes:
        if run_episode(healthy_fn, ep):
            healthy_pass.add(ep)
    results = {}
    for case_id, case in CASES.items():
        broken = case["plant"](healthy)
        broken_fn = make_policy_fn(broken)
        broken_pass = set()
        for ep in episodes:
            if run_episode(broken_fn, ep):
                broken_pass.add(ep)
        target = sorted(healthy_pass - broken_pass)
        breakage = sorted(healthy_pass & broken_pass)
        results[case_id] = {
            "case": case,
            "healthy": healthy,
            "broken": broken,
            "episodes_total": len(episodes),
            "healthy_pass": len(healthy_pass),
            "broken_pass": len(broken_pass),
            "target": target,
            "breakage": breakage,
        }
    return results
