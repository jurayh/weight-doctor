"""LLM-doctor interface and harness for kill 3.

Kill 3 says the LLM doctor must beat the scripted baseline on the
toy catalog, or the LLM-doctor claim goes. The precise bar is
pre-registered in KILL3_PLAN.md (fix 4/4 with zero regressions
AND strictly fewer total weights changed than the scripted
doctor's 41); until that run happens, kill 3 is UNTESTED.

Fairness rule: the LLM receives exactly the weight evidence the
scripted doctor uses (W1 group norms, Wout row norms and biases,
Wout entries above the schema maximum), plus the schema diagram.
It does NOT see episode outcomes, the catalog ground truth, or the
scripted doctor's diff. The comparison then runs the same verifier
(`weight_doctor/verifier.py`) on the LLM's diffs.

Running the comparison against a paid API needs an estimate-first
approval (standing rule: present the estimated cost from measured
tokens, get explicit approval, smoke tests included). The client
is injected; with no client this module refuses to run.
"""

import json
import math

import numpy as np

from .doctor import INPUT_GROUPS, SCHEMA_MAX_WOUT
from .game import ACTIONS, NCELL
from .policy import build_schema_weights

PROMPT_TEMPLATE = """You are the Weight Doctor for a toy 3x3 grid MLP policy.
Schema: 27 inputs (agent 0-8, goal 9-17, hazard 18-26) -> 128 ReLU
-> 128 ReLU -> 4 outputs (RIGHT, DOWN, LEFT, UP). Healthy Wout
entries are only 0, 1, or 2. Healthy W1 has agent, goal, and hazard
pass-through units plus a goal_present unit.
Wout wiring (the schema the healthy policy is built from): hidden
units 0-35 are safe-reducing units, one per (cell, direction):
unit c*4+d serves cell c in direction d. Hidden units 36-71 are
fallback units, same layout. For each cell c and direction d where
the neighbour in direction d exists, Wout[d, c*4+d] = 2.0 and
Wout[d, 36+c*4+d] = 1.0; every other Wout entry is 0. Directions:
0=RIGHT, 1=DOWN, 2=LEFT, 3=UP; cells are row-major, so from cell c
RIGHT exists unless c is in column 2 (c = 2, 5, 8), DOWN exists
unless c is in row 2 (c = 6, 7, 8), LEFT exists unless c is in
column 0 (c = 0, 3, 6), UP exists unless c is in row 0 (c = 0, 1, 2).
Weight evidence (JSON): {evidence}
Return ONLY JSON: {{"cause": "...", "diff": [[matrix, row, col, value], ...]}}
For bias vectors (b1, b2, bout) the entry is [matrix, row, null, value],
e.g. ["bout", 0, null, 0.0]. For matrices (W1, W2, Wout) col is the
column index, e.g. ["Wout", 0, 4, 2.0].
Edits must be surgical: a diff touching a large fraction of weights
does not count as a fix.
"""

# Matrices the doctor may touch, with their shapes, for validation.
# Shapes are (rows, cols); bias vectors have cols=None.
MATRIX_SHAPES = {
    "W1": (128, 27), "b1": (128, None),
    "W2": (128, 128), "b2": (128, None),
    "Wout": (4, 128), "bout": (4, None),
}


class LLMDoctorNotRun(RuntimeError):
    pass


class LLMDoctorBadDiff(ValueError):
    pass


def build_evidence(weights):
    """The same weight evidence the scripted doctor uses, as JSON-able data."""
    schema = build_schema_weights()
    evidence = {"w1_group_norms": {}, "wout_rows": [], "wout_above_schema_max": []}
    for name, lo, hi in INPUT_GROUPS:
        evidence["w1_group_norms"][name] = float(
            np.linalg.norm(weights["W1"][:, lo:hi]))
    for a, action in enumerate(ACTIONS):
        evidence["wout_rows"].append({
            "action": action,
            "row_norm": float(np.linalg.norm(weights["Wout"][a])),
            "bias": float(weights["bout"][a]),
        })
    for a in range(len(ACTIONS)):
        for col in range(weights["Wout"].shape[1]):
            val = float(weights["Wout"][a, col])
            if val > SCHEMA_MAX_WOUT:
                evidence["wout_above_schema_max"].append({
                    "action": ACTIONS[a], "unit": col, "value": val,
                    "schema_value": float(schema["Wout"][a, col]),
                })
    return evidence


def build_prompt(weights):
    """The exact prompt string the LLM doctor receives for a policy."""
    evidence = json.dumps(build_evidence(weights), separators=(",", ":"))
    return PROMPT_TEMPLATE.format(evidence=evidence)


def parse_diff(text):
    """Parse and validate an LLM diff. Returns a list of (matrix, row, col, value).

    Raises LLMDoctorBadDiff on any malformed, out-of-range, or
    non-finite entry, or on duplicate entries. The caller records
    the failure; a bad diff never silently becomes a partial edit.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMDoctorBadDiff(f"response is not JSON: {e}")
    if not isinstance(obj, dict) or "diff" not in obj:
        raise LLMDoctorBadDiff("response JSON has no 'diff' key")
    raw = obj["diff"]
    if not isinstance(raw, list):
        raise LLMDoctorBadDiff("'diff' is not a list")
    diff = []
    seen = set()
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != 4:
            raise LLMDoctorBadDiff(f"diff entry is not [matrix, row, col, value]: {entry!r}")
        matrix, row, col, value = entry
        if matrix not in MATRIX_SHAPES:
            raise LLMDoctorBadDiff(f"unknown matrix: {matrix!r}")
        rows, cols = MATRIX_SHAPES[matrix]
        if not isinstance(row, int) or not 0 <= row < rows:
            raise LLMDoctorBadDiff(f"row out of range for {matrix}: {row!r}")
        if cols is None:
            if col is not None:
                raise LLMDoctorBadDiff(f"bias entry for {matrix} must have col null, got {col!r}")
        else:
            if not isinstance(col, int) or not 0 <= col < cols:
                raise LLMDoctorBadDiff(f"col out of range for {matrix}: {col!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise LLMDoctorBadDiff(f"non-finite value: {value!r}")
        key = (matrix, row, col)
        if key in seen:
            raise LLMDoctorBadDiff(f"duplicate diff entry: {key}")
        seen.add(key)
        diff.append((matrix, row, col, float(value)))
    return diff, obj.get("cause", "")


def run_llm_doctor(weights, client=None):
    """Run the LLM doctor on one policy. Returns (diff, cause, raw_text, usage).

    `client` must have `.complete(prompt) -> {"text": str, "usage": dict}`.
    With no client, refuses: the $0 core runs the scripted doctor only.
    """
    if client is None:
        raise LLMDoctorNotRun(
            "LLM doctor not run: no estimate-first approval has been "
            "given for paid API use, and the $0 core runs the scripted "
            "doctor only. See KILLS.md kill 3 and KILL3_PLAN.md."
        )
    prompt = build_prompt(weights)
    result = client.complete(prompt)
    diff, cause = parse_diff(result["text"])
    return diff, cause, result["text"], result.get("usage", {})
