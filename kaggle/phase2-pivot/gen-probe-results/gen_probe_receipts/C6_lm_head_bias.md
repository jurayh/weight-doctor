# Receipt — C6_lm_head_bias
Family: lm_head </think> bias
Scope: Qwen3.5-0.8B, thinking mode ON, Kaggle T4, 1024-token cap, HELD-OUT battery: these prompts were written and frozen before this run and were never used in candidate selection. No claim about other models, modes, or batteries.
Diff: 1024 weights changed (1.360990821733084e-06), L2=0.0866, layer-limited=True, edit_type=weight edit
Note: Alpha targets a +5 logit increase on average hidden alignment; actual per-token increase varies. If lm_head is tied, this row edit also changes the input embedding for that token.
Target: 5/9
- PASS: Emma has 31 marbles. She loses 9 and buys 12 more. How many marbles do (closed=True, correct=True, tokens=622)
- fail: A farmer packs 6 boxes with 15 eggs each. He sells 47 eggs. How many e (closed=False, correct=False, tokens=1024)
- PASS: Lily reads 24 pages a day. How many pages does she read in 3 weeks? (closed=True, correct=True, tokens=745)
- PASS: A garden is 7 m long and 6 m wide. What is its area in square m? (closed=True, correct=True, tokens=739)
- fail: A bus travels 45 km per hour for 2 hours, then travels 30 km more. How (closed=False, correct=False, tokens=1024)
- PASS: Mia earns $12 per hour and works 4 hours a day, 6 days a week. How muc (closed=True, correct=True, tokens=783)
- PASS: A store sells pencils for $2 each. Tom buys 9 pencils and pays with $2 (closed=True, correct=True, tokens=585)
- fail: What is the capital of Italy? Answer in one sentence. (closed=False, correct=False, tokens=1024)
- fail: What colour is a ripe banana? Answer in one sentence. (closed=False, correct=False, tokens=1024)

Breakage regressions: 0 at indices []
Kill P1: True, Kill P2: True
