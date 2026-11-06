"""The toy game: a 3x3 grid, one agent, one goal, one hazard.

Everything is enumerable. An episode is a triple (agent, goal, hazard)
of distinct cells, so there are exactly 9 * 8 * 7 = 504 episodes.
No sampling, no lucky runs: a battery is all 504, or a defined subset
of them, and the result is a count, not an estimate.

Actions are ordered [RIGHT, DOWN, LEFT, UP] and the policy breaks
ties by taking the earliest action in that order (numpy argmax).
An invalid move (off the grid) stays in place and counts as stuck.
Stepping onto the hazard fails. Revisiting a cell fails: the policy
is a deterministic function of the agent cell (goal and hazard are
fixed within an episode), so a revisit guarantees an infinite loop.
"""

GRID = 3
NCELL = GRID * GRID
ACTIONS = ["RIGHT", "DOWN", "LEFT", "UP"]
# (dx, dy) per action, in ACTIONS order.
DELTAS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
MAX_STEPS = 20


def cell_xy(cell):
    return cell % GRID, cell // GRID


def cell_at(x, y):
    if 0 <= x < GRID and 0 <= y < GRID:
        return y * GRID + x
    return None


def neighbour(cell, action):
    """Cell reached by taking `action` from `cell`, or None off-grid."""
    x, y = cell_xy(cell)
    dx, dy = DELTAS[action]
    return cell_at(x + dx, y + dy)


def all_episodes():
    """All 504 distinct (agent, goal, hazard) triples, in fixed order."""
    eps = []
    for agent in range(NCELL):
        for goal in range(NCELL):
            if goal == agent:
                continue
            for hazard in range(NCELL):
                if hazard in (agent, goal):
                    continue
                eps.append((agent, goal, hazard))
    return eps


def run_episode(policy, episode):
    """Run one episode. Returns True on reaching the goal.

    `policy` is called as policy(agent, goal, hazard) -> action index.
    """
    agent, goal, hazard = episode
    visited = {agent}
    pos = agent
    for _ in range(MAX_STEPS):
        action = int(policy(pos, goal, hazard))
        nxt = neighbour(pos, action)
        if nxt is None:
            return False  # bumped into a wall and stayed: stuck
        if nxt == hazard:
            return False
        if nxt == goal:
            return True
        if nxt in visited:
            return False  # deterministic loop
        visited.add(nxt)
        pos = nxt
    return False
