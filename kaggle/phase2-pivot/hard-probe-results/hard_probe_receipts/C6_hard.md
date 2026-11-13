# Receipt — C6_hard
Scope: Qwen3.5-0.8B, thinking mode ON, Kaggle T4, 1024-token cap, HARD battery: 9 multi-step problems frozen Nov 7, 2026 in C6_HARD_PROBE_PLAN.md, no overlap with the original 23 or the held-out 17. Control: C6 on the original 9 target prompts. No claim about other models, modes, or batteries.
Diff: 1024 weights changed (1.360990821733084e-06), L2=0.0866, layer-limited=True, edit_type=weight edit
Note: Alpha targets a +5 logit increase on average hidden alignment; actual per-token increase varies. If lm_head is tied, this row edit also changes the input embedding for that token.
Target: 0/9
- fail: A bakery bakes 240 loaves in the morning. They sell 3/8 of them before (closed=False, correct=False, tokens=1024)
- fail: A train travels 120 km in 2 hours, then 180 km in 3 hours. What is its (closed=False, correct=False, tokens=1024)
- fail: Sara has 4 shelves with 18 books each. She gives away 25 books and buy (closed=False, correct=False, tokens=1024)
- fail: Tom earns $15 per hour. He works 6 hours a day for 5 days, then spends (closed=False, correct=False, tokens=1024)
- fail: A farmer has 96 apples. He packs 1/4 of them into boxes, gives 18 to a (closed=False, correct=False, tokens=1024)
- fail: A car drives 90 km in 1.5 hours, stops for 30 minutes, then drives 60  (closed=False, correct=False, tokens=1024)
- fail: A shop sells pens in packs of 8 for $6 and packs of 12 for $8. Maria n (closed=False, correct=False, tokens=1024)
- fail: Lily reads 15 pages on Monday, twice as many on Tuesday as Monday, and (closed=False, correct=False, tokens=1024)
- fail: A farmer collects 5 dozen eggs each morning. He sells 3 dozen at the m (closed=False, correct=False, tokens=1024)

Verdict: FORCING CLOSURE HURTS: C6 scores below thinking-off on hard problems (H=0/9, thinking-off=7/9). The generalisation claim narrows to easy problems.
