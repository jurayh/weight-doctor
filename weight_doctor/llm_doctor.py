"""LLM-doctor interface: defined, not run, in the $0 core.

Kill 3 says the LLM doctor must beat the scripted baseline on the
toy catalog, or the LLM-doctor claim goes. Running that comparison
against a paid API needs an estimate-first approval (standing rule:
present the estimated cost from measured tokens, get explicit
approval, smoke tests included), and no such approval has been
given for this build. So this module defines the interface the
comparison will use, and refuses to run without a model client.

The interface: the LLM receives the same weight evidence the
scripted doctor uses (W1 group norms, Wout row norms and biases,
Wout entries above the schema maximum), plus the schema diagram,
and must return a JSON diff in exactly the format apply_diff
accepts, with a one-line cause per fault. The comparison then runs
the same verifier on the LLM's diffs. Until that run happens,
kill 3 is UNTESTED, not passed and not failed.
"""

PROMPT_TEMPLATE = """You are the Weight Doctor for a toy 3x3 grid MLP policy.
Schema: 27 inputs (agent 0-8, goal 9-17, hazard 18-26) -> 128 ReLU
-> 128 ReLU -> 4 outputs (RIGHT, DOWN, LEFT, UP). Healthy Wout
entries are only 0, 1, or 2. Healthy W1 has agent, goal, and hazard
pass-through units plus a goal_present unit.
Weight evidence (JSON): {evidence}
Return ONLY JSON: {{"cause": "...", "diff": [[matrix, row, col, value], ...]}}
Edits must be surgical: a diff touching a large fraction of weights
does not count as a fix.
"""


class LLMDoctorNotRun(RuntimeError):
    pass


def run_llm_doctor(weights, client=None):
    """Placeholder: requires an approved, costed model client."""
    raise LLMDoctorNotRun(
        "LLM doctor not run: no estimate-first approval has been "
        "given for paid API use, and the $0 core runs the scripted "
        "doctor only. See KILLS.md kill 3."
    )
