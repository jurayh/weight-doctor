"""Frozen batteries for the Phase 2 pivot (see PHASE2_PIVOT_PLAN.md).

Target: the 9 cap-sensitivity gate prompts, verbatim from the
loop hunt / cap-sensitivity probes. Breakage: the 4 prompts
that already closed at 1024 tokens, plus 10 NEW short-QA
prompts written at build time (Nov 4, 2026) and frozen here.
Changing any prompt, expected answer, or success rule after
seeing candidate-edit results would be moving the goalposts.

Success on a prompt = BOTH:
  (a) thinking closes: "</think>" appears in the output AND
      generation stopped before the cap, AND
  (b) the text after the FIRST "</think>" contains the
      expected answer (number, or "Paris"/"blue"/"Tokyo"/
      "green"/"Earth"/"January", case-insensitive).
Closing with a wrong answer is NOT a success. Answering
correctly inside unclosed thinking is NOT a success.
For numeric answers, matching is on the standalone number
as a word boundary, so "32" does not match "132" or "320".
"""

import re

# (prompt, expected) — target, base rate 0/9 at 1024 tokens.
TARGET = [
    ("Maria has 24 apples. She gives 7 to her brother and buys 15 more. How many apples does she have now?", "32"),
    ("A baker makes 5 trays with 12 muffins each. He sells 38 muffins. How many muffins are left?", "22"),
    ("Tom reads 18 pages a day. How many pages does he read in 2 weeks?", "252"),
    ("A rectangle is 9 cm long and 4 cm wide. What is its area in square cm?", "36"),
    ("A train travels 60 km per hour for 3 hours, then travels 40 km more. How far did it travel in total, in km?", "220"),
    ("Jake earns $8 per hour and works 6 hours a day, 5 days a week. How much does he earn in a week, in dollars?", "240"),
    ("A store sells notebooks for $3 each. Ana buys 7 notebooks and pays with $25. How much change does she get, in dollars?", "4"),
    ("What is the capital of France? Answer in one sentence.", "Paris"),
    ("What colour is the sky on a clear day? Answer in one sentence.", "blue"),
]

# (prompt, expected) — known breakage, base rate 4/4 at 1024 tokens.
BREAKAGE_KNOWN = [
    ("Sara has 3 boxes with 14 pencils each, plus 5 loose pencils. How many pencils does she have in total?", "47"),
    ("There are 48 students. Each table seats 6 students. How many tables are needed?", "8"),
    ("A garden has 5 rows with 11 plants each. 9 plants die. How many plants are left?", "46"),
    ("What is 2 + 3? Answer with just the number.", "5"),
]

# (prompt, expected) — NEW breakage, frozen at build time.
# Style matches the known breakage: one fact, one short
# calculation, or one constrained instruction. None asks
# for repetition. Baseline pass rate is measured, not assumed.
BREAKAGE_NEW = [
    ("What is the capital of Japan? Answer in one sentence.", "Tokyo"),
    ("What is 7 + 8? Answer with just the number.", "15"),
    ("How many days are in a week? Answer with just the number.", "7"),
    ("What colour is grass on a sunny day? Answer in one sentence.", "green"),
    ("A box has 6 pens. Tom buys 4 boxes. How many pens does he have?", "24"),
    ("What is 10 - 4? Answer with just the number.", "6"),
    ("Name the planet we live on. Answer in one sentence.", "Earth"),
    ("Sara has 5 apples and eats 2. How many apples are left?", "3"),
    ("What is the first month of the year? Answer in one word.", "January"),
    ("A car travels 30 km in 1 hour. How far does it travel in 2 hours at the same speed, in km?", "60"),
]

BREAKAGE = BREAKAGE_KNOWN + BREAKAGE_NEW

BANNED_PHRASES = [
    "repeat", "again, then again", "list them again",
    "count from", "one number per line", "50 times",
    "write it again", "then write it again",
]


def check_guard(prompts):
    bad = [p for p in prompts if any(b in p.lower() for b in BANNED_PHRASES)]
    if bad:
        raise ValueError(f"prompt guard failed: {bad}")


def check_success(output, new_tokens, max_tokens, expected):
    """Return (success, closed, correct, reason).

    closed: "</think>" present and stopped before the cap.
    correct: expected answer appears in the text after the
    FIRST "</think>" (or, if never closed, False by
    definition — correct-in-thinking is not a success).
    """
    closed = ("</think>" in output) and (new_tokens < max_tokens)
    if not closed:
        return False, False, False, "did not close thinking"
    after = output.split("</think>", 1)[1]
    if expected.isdigit():
        correct = re.search(rf"\b{re.escape(expected)}\b", after) is not None
    else:
        correct = expected.lower() in after.lower()
    if not correct:
        return False, True, False, "closed but wrong final answer"
    return True, True, True, "closed with correct final answer"


def self_check_markers(output):
    """Counts of self-check markers, for diagnosis only."""
    low = output.lower()
    return {m: low.count(m) for m in
            ("wait", "okay", "final answer", "check", "confirm")}
