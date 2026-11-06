# Receipt — C_direction_0.5
Family: direction subtraction at L15
Scope: Qwen3.5-0.8B, thinking mode ON, Kaggle T4, 1024-token cap, HELD-OUT battery: these prompts were written and frozen before this run and were never used in candidate selection. No claim about other models, modes, or batteries.
Diff: 0 weights changed (0.0), L2=None, layer-limited=True, edit_type=hook intervention (not a weight edit)
Note: Hook changes no weights, so the P2 weight count passes trivially. A P1 pass here still needs a weight-edit conversion before claiming a surgical weight fix.
Target: 2/9
- fail: Emma has 31 marbles. She loses 9 and buys 12 more. How many marbles do (closed=False, correct=False, tokens=1024)
- fail: A farmer packs 6 boxes with 15 eggs each. He sells 47 eggs. How many e (closed=False, correct=False, tokens=1024)
- fail: Lily reads 24 pages a day. How many pages does she read in 3 weeks? (closed=False, correct=False, tokens=1024)
- fail: A garden is 7 m long and 6 m wide. What is its area in square m? (closed=False, correct=False, tokens=1024)
- fail: A bus travels 45 km per hour for 2 hours, then travels 30 km more. How (closed=False, correct=False, tokens=1024)
- fail: Mia earns $12 per hour and works 4 hours a day, 6 days a week. How muc (closed=False, correct=False, tokens=1024)
- PASS: A store sells pencils for $2 each. Tom buys 9 pencils and pays with $2 (closed=True, correct=True, tokens=648)
- PASS: What is the capital of Italy? Answer in one sentence. (closed=True, correct=True, tokens=960)
- fail: What colour is a ripe banana? Answer in one sentence. (closed=False, correct=False, tokens=1024)

Breakage regressions: 0 at indices []
Kill P1: False, Kill P2: True
