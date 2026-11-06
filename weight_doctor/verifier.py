"""The verifier: two batteries plus minimality, nothing else.

For a case, given the broken and edited policies:
  target battery   (before): broken fails every target episode by
                     construction; (after): edited must pass all
                     of them.
  breakage battery (after): edited must pass every episode in the
                     breakage battery, i.e. zero regressions vs
                     the broken policy on episodes the broken
                     policy already passed (which, by construction,
                     are exactly the healthy-passing non-target
                     episodes).
  minimality: fraction of weights changed and edit L2 norm, against
                     the pre-registered bars in KILLS.md.

A fix counts only if target is fully green AND breakage has zero
regressions AND minimality is inside both bars. A fix that breaks
something else is not a fix, and an edit that rewrites the network
does not count as a fix.
"""

from .game import run_episode
from .policy import diff_stats, make_policy_fn

# Pre-registered bars (KILLS.md). Changing these after seeing
# results would be moving the goalposts; they live here, in code.
KILL2_MAX_FRACTION = 0.02
KILL2_MAX_L2 = 20.0


def verify_case(broken, edited, target, breakage):
    broken_fn = make_policy_fn(broken)
    edited_fn = make_policy_fn(edited)
    target_before_pass = [ep for ep in target
                          if run_episode(broken_fn, ep)]
    target_after_fail = [ep for ep in target
                         if not run_episode(edited_fn, ep)]
    breakage_after_fail = [ep for ep in breakage
                           if not run_episode(edited_fn, ep)]
    stats = diff_stats(broken, edited)
    minimal = (stats["fraction_changed"] <= KILL2_MAX_FRACTION
               and stats["l2_norm"] <= KILL2_MAX_L2)
    fixed = (not target_after_fail and not breakage_after_fail
             and minimal)
    return {
        "target_total": len(target),
        "target_before_pass": len(target_before_pass),
        "target_after_fail": target_after_fail,
        "breakage_total": len(breakage),
        "breakage_after_fail": breakage_after_fail,
        "minimality": stats,
        "minimality_within_bars": minimal,
        "fixed": fixed,
    }
