# Receipt — C_direction_0.5
Family: direction subtraction at L15
Scope: Qwen3.5-0.8B, thinking mode ON, Kaggle T4, 1024-token cap, this battery only. No claim about other models, modes, or batteries.
Diff: 0 weights changed (0.0), L2=None, layer-limited=True, edit_type=hook intervention (not a weight edit)
Note: Hook changes no weights, so the P2 weight count passes trivially. A P1 pass here still needs a weight-edit conversion before claiming a surgical weight fix.
Target: 5/9
- PASS: Maria has 24 apples. She gives 7 to her brother and buys 15 more. How  (closed=True, correct=True, tokens=751)
- fail: A baker makes 5 trays with 12 muffins each. He sells 38 muffins. How m (closed=False, correct=False, tokens=1024)
- fail: Tom reads 18 pages a day. How many pages does he read in 2 weeks? (closed=False, correct=False, tokens=1024)
- PASS: A rectangle is 9 cm long and 4 cm wide. What is its area in square cm? (closed=True, correct=True, tokens=467)
- fail: A train travels 60 km per hour for 3 hours, then travels 40 km more. H (closed=False, correct=False, tokens=1024)
- PASS: Jake earns $8 per hour and works 6 hours a day, 5 days a week. How muc (closed=True, correct=True, tokens=613)
- fail: A store sells notebooks for $3 each. Ana buys 7 notebooks and pays wit (closed=False, correct=False, tokens=1024)
- PASS: What is the capital of France? Answer in one sentence. (closed=True, correct=True, tokens=194)
- PASS: What colour is the sky on a clear day? Answer in one sentence. (closed=True, correct=True, tokens=253)

Breakage regressions: 1 at indices [0]
Kill P1: False, Kill P2: True
