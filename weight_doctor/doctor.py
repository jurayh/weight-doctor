"""The scripted doctor: read weights, name the cause, propose a diff.

Tools, in the brief's sense: read weights by layer, head, and
channel. Here the channels are the three input groups (agent /
goal / hazard) and the output rows (one per action).

Diagnosis rules (all thresholds are on the weights themselves,
not on episode outcomes):
  * an input group's W1 column norm is 0 -> that channel is blind.
  * an output row is all zero with a negative bias -> that action
    is dead.
  * any Wout entry above the schema maximum (2) -> a spurious
    weight. Healthy Wout entries are only 0, 1, or 2.

Repairs use the published schema (policy.build_schema_weights) to
compute the correct value for exactly the diagnosed entries, and
nothing else. The repair is returned as a diff; the doctor never
returns a rewritten model. If no rule fires, the doctor proposes
no diff and says so: an honest no-diagnosis beats a guessed edit.

This is the scripted baseline. The LLM-doctor layer (kill 3) is a
separate interface, not run in the $0 core: see llm_doctor.py.
Its claim is untested until an estimate-first approval funds it.
"""

import numpy as np

from .game import ACTIONS, NCELL, neighbour
from .policy import GOAL_PRESENT_UNIT, build_schema_weights

SCHEMA_MAX_WOUT = 2.0
INPUT_GROUPS = [("agent", 0, NCELL), ("goal", NCELL, 2 * NCELL),
                ("hazard", 2 * NCELL, 3 * NCELL)]


def diagnose(weights):
    """Return a list of diagnosis dicts, each with a diff."""
    schema = build_schema_weights()
    diagnoses = []

    # Channel blindness: W1 column-group norms.
    for name, lo, hi in INPUT_GROUPS:
        norm = float(np.linalg.norm(weights["W1"][:, lo:hi]))
        if norm == 0.0:
            diff = []
            if name == "goal":
                for g in range(NCELL):
                    diff.append(("W1", NCELL + g, NCELL + g, 1.0))
                for g in range(NCELL):
                    diff.append(("W1", GOAL_PRESENT_UNIT, NCELL + g, 1.0))
            elif name == "hazard":
                for h in range(NCELL):
                    diff.append(("W1", 2 * NCELL + h, 2 * NCELL + h, 1.0))
            elif name == "agent":
                for c in range(NCELL):
                    diff.append(("W1", c, c, 1.0))
            diagnoses.append({
                "fault": f"blind_{name}",
                "cause": (f"W1 columns for the {name} input group "
                          f"have norm 0: the {name} channel never "
                          f"reaches layer 1."),
                "evidence": {"w1_group_norm": norm,
                             "group": name},
                "diff": diff,
            })

    # Dead actions: zero output row + negative bias.
    for a, action in enumerate(ACTIONS):
        row = weights["Wout"][a]
        if float(np.linalg.norm(row)) == 0.0 and weights["bout"][a] < 0:
            diff = []
            for col in range(weights["Wout"].shape[1]):
                val = float(schema["Wout"][a, col])
                if val != 0.0:
                    diff.append(("Wout", a, col, val))
            diff.append(("bout", a, None, 0.0))
            diagnoses.append({
                "fault": f"dead_action_{action.lower()}",
                "cause": (f"Output row for {action} is zero and its "
                          f"bias is {weights['bout'][a]:g}, so "
                          f"{action} can never win argmax."),
                "evidence": {"wout_row_norm": 0.0,
                             "bias": float(weights["bout"][a]),
                             "action": action},
                "diff": diff,
            })

    # Spurious weights: Wout entries above anything the schema uses.
    spurious = []
    for a in range(len(ACTIONS)):
        for col in range(weights["Wout"].shape[1]):
            val = float(weights["Wout"][a, col])
            if val > SCHEMA_MAX_WOUT:
                spurious.append((a, col, val,
                                 float(schema["Wout"][a, col])))
    if spurious:
        diff = [("Wout", a, col, schema_val)
                for a, col, _, schema_val in spurious]
        diagnoses.append({
            "fault": "spurious_weight",
            "cause": ("Wout entries above the schema maximum 2: "
                      "a spurious connection dominates its action "
                      "wherever its hidden unit fires."),
            "evidence": {"entries": [
                {"action": ACTIONS[a], "unit": col, "value": val,
                 "schema_value": schema_val}
                for a, col, val, schema_val in spurious]},
            "diff": diff,
        })

    return diagnoses


def propose_repair(weights):
    """All diagnoses for a policy, with one combined diff."""
    diagnoses = diagnose(weights)
    diff = []
    for d in diagnoses:
        diff.extend(d["diff"])
    return diagnoses, diff
