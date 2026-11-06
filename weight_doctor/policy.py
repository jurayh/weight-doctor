"""Toy MLP policies, built from a published wiring schema.

Architecture: ReLU MLP, 27 inputs -> 128 -> 128 -> 4 outputs.
Inputs are three one-hot groups: agent cells 0-8, goal cells 9-17,
hazard cells 18-26.

Layer 1 (the wiring spec, not a learned accident):
  units 0-8    agent pass-through
  units 9-17   goal pass-through
  units 18-26  hazard pass-through
  unit 27      goal_present = 1 iff any goal input is on
  units 28+    unused (zero)

Layer 2:
  units c*4+d (0-35)      safe_reducing[c][d]: agent at cell c, the
                          goal lies in direction d (so moving d
                          shortens Manhattan distance), and the
                          neighbour in direction d is not the hazard.
                          Its value is the distance along that axis
                          (1 or 2), so among reducing moves the
                          larger-axis move wins: without this, a tie
                          between two reducing moves can walk into
                          a trap corner and loop.
  units 36+c*4+d (36-71)  fallback_safe[c][d]: agent at cell c, a
                          goal is present at all, and the neighbour
                          in direction d is not the hazard. This is
                          the hazard-detour move: it only fires when
                          the goal channel is live, so a goal-blind
                          policy gets no fallback and stalls instead
                          of accidentally succeeding by wandering.
  units 72+             unused (zero)

Output: action d scores 2 * safe_reducing + 1 * fallback_safe, so a
reducing safe move (3 or 5) always beats a detour move (1), which
always beats an unsafe or invalid move (0).

Total parameters: W1 128x27 + b1 128 + W2 128x128 + b2 128
+ Wout 4x128 + bout 4 = 20,612. Tens of thousands, as promised, and
most of them are structural zeros: the circuit lives in a few hundred
named entries. That is the point of a toy: the wiring is readable.

A policy is a dict of numpy arrays. It is never rewritten in place
by the doctor: edits are diffs (see doctor.py), applied by
apply_diff, and the receipt records exactly which entries moved.
"""

import hashlib

import numpy as np

from .game import ACTIONS, NCELL, neighbour

SCHEMA_VERSION = "1"
HIDDEN = 128
INPUT_DIM = 3 * NCELL  # 27
GOAL_PRESENT_UNIT = 27


def _goal_distance_in_direction(cell, direction, goal):
    """Distance along `direction`'s axis to `goal`, or 0 if moving
    that way would not shorten the Manhattan distance."""
    from .game import cell_xy

    cx, cy = cell_xy(cell)
    gx, gy = cell_xy(goal)
    if direction == 0:  # RIGHT
        return gx - cx if gx > cx else 0
    if direction == 1:  # DOWN
        return gy - cy if gy > cy else 0
    if direction == 2:  # LEFT
        return cx - gx if gx < cx else 0
    return cy - gy if gy < cy else 0  # UP


def build_schema_weights():
    """The spec itself: matrices the healthy policy is built from.

    The doctor uses this same function to compute repair values.
    That is disclosed, not hidden: on toys the doctor is
    schema-guided (it checks weights against the published wiring
    diagram and repairs deviations). It does not copy a healthy
    model wholesale; a repair touches only the diagnosed entries.
    """
    W1 = np.zeros((HIDDEN, INPUT_DIM))
    b1 = np.zeros(HIDDEN)
    for c in range(NCELL):
        W1[c, c] = 1.0  # agent pass-through
        W1[NCELL + c, NCELL + c] = 1.0  # goal pass-through
        W1[2 * NCELL + c, 2 * NCELL + c] = 1.0  # hazard pass-through
    for g in range(NCELL):
        W1[GOAL_PRESENT_UNIT, NCELL + g] = 1.0
    # bias 0: goal_present is exactly 1 iff any goal input is on.

    W2 = np.zeros((HIDDEN, HIDDEN))
    b2 = np.zeros(HIDDEN)
    for c in range(NCELL):
        for d in range(len(ACTIONS)):
            nxt = neighbour(c, d)
            if nxt is None:
                continue
            sr = c * 4 + d
            fb = 36 + c * 4 + d
            # safe_reducing: 2*agent + dist - 3*hazard - 2.
            # Value = dist (1 or 2) when agent is at c, the goal is
            # dist away along d, and the neighbour is not the hazard;
            # 0 in every other case (the -3 hazard weight is what
            # keeps a dist-2 move from firing through a hazard).
            W2[sr, c] = 2.0
            for g in range(NCELL):
                dist = _goal_distance_in_direction(c, d, g)
                if dist:
                    W2[sr, NCELL + g] = float(dist)
            W2[sr, 2 * NCELL + nxt] = -3.0
            b2[sr] = -2.0
            # fallback_safe: agent_c + goal_present - hazard_nbr - 1
            W2[fb, c] = 1.0
            W2[fb, GOAL_PRESENT_UNIT] = 1.0
            W2[fb, 2 * NCELL + nxt] = -1.0
            b2[fb] = -1.0

    Wout = np.zeros((len(ACTIONS), HIDDEN))
    bout = np.zeros(len(ACTIONS))
    for c in range(NCELL):
        for d in range(len(ACTIONS)):
            if neighbour(c, d) is None:
                continue
            Wout[d, c * 4 + d] = 2.0
            Wout[d, 36 + c * 4 + d] = 1.0
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "Wout": Wout, "bout": bout}


def healthy_policy():
    return build_schema_weights()


def forward(weights, agent, goal, hazard):
    x = np.zeros(INPUT_DIM)
    x[agent] = 1.0
    x[NCELL + goal] = 1.0
    x[2 * NCELL + hazard] = 1.0
    h1 = np.maximum(0.0, weights["W1"] @ x + weights["b1"])
    h2 = np.maximum(0.0, weights["W2"] @ h1 + weights["b2"])
    return weights["Wout"] @ h2 + weights["bout"]


def make_policy_fn(weights):
    def policy(agent, goal, hazard):
        return int(np.argmax(forward(weights, agent, goal, hazard)))

    return policy


def total_params(weights):
    return sum(int(v.size) for v in weights.values())


def weights_hash(weights):
    h = hashlib.sha256()
    for key in ("W1", "b1", "W2", "b2", "Wout", "bout"):
        h.update(weights[key].tobytes())
    return h.hexdigest()


def copy_weights(weights):
    return {k: v.copy() for k, v in weights.items()}


def apply_diff(weights, diff):
    """Return a new weights dict with `diff` applied.

    A diff is a list of (matrix_name, row, col, new_value) entries;
    for 1-D bias vectors col is None and only row is used.
    The original weights are untouched. The diff, not a rewritten
    model, is what the receipt records.
    """
    out = copy_weights(weights)
    for name, row, col, value in diff:
        if out[name].ndim == 1:
            out[name][row] = value
        else:
            out[name][row, col] = value
    return out


def diff_stats(before, after):
    """Minimality stats for an edit: count, fraction, L2, Linf."""
    changed = 0
    sq = 0.0
    linf = 0.0
    entries = []
    for name in ("W1", "b1", "W2", "b2", "Wout", "bout"):
        d = after[name] - before[name]
        mask = d != 0
        changed += int(mask.sum())
        sq += float((d[mask] ** 2).sum()) if mask.any() else 0.0
        if mask.any():
            linf = max(linf, float(np.abs(d[mask]).max()))
            idx = np.argwhere(mask)
            for pos in idx:
                if d.ndim == 1:
                    r = int(pos[0])
                    entries.append((name, r, None,
                                    float(before[name][r]),
                                    float(after[name][r])))
                else:
                    r, c = int(pos[0]), int(pos[1])
                    entries.append((name, r, c,
                                    float(before[name][r, c]),
                                    float(after[name][r, c])))
    total = total_params(before)
    return {
        "changed": changed,
        "total_params": total,
        "fraction_changed": changed / total,
        "l2_norm": float(sq ** 0.5),
        "linf": linf,
        "entries": entries,
    }
