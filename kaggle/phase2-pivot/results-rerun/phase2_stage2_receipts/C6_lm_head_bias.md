# Receipt — C6_lm_head_bias
Family: lm_head </think> bias
Scope: Qwen3.5-0.8B, thinking mode ON, Kaggle T4, 1024-token cap, this battery only. No claim about other models, modes, or batteries.
Diff: 1024 weights changed (1.360990821733084e-06), L2=0.0866, layer-limited=True, edit_type=weight edit
Note: Alpha targets a +5 logit increase on average hidden alignment; actual per-token increase varies. If lm_head is tied, this row edit also changes the input embedding for that token.
Target: 5/9
- PASS: Maria has 24 apples. She gives 7 to her brother and buys 15 more. How  (closed=True, correct=True, tokens=466)
- PASS: A baker makes 5 trays with 12 muffins each. He sells 38 muffins. How m (closed=True, correct=True, tokens=778)
- fail: Tom reads 18 pages a day. How many pages does he read in 2 weeks? (closed=False, correct=False, tokens=1024)
- PASS: A rectangle is 9 cm long and 4 cm wide. What is its area in square cm? (closed=True, correct=True, tokens=414)
- fail: A train travels 60 km per hour for 3 hours, then travels 40 km more. H (closed=False, correct=False, tokens=1024)
- fail: Jake earns $8 per hour and works 6 hours a day, 5 days a week. How muc (closed=False, correct=False, tokens=1024)
- PASS: A store sells notebooks for $3 each. Ana buys 7 notebooks and pays wit (closed=True, correct=True, tokens=878)
- PASS: What is the capital of France? Answer in one sentence. (closed=True, correct=True, tokens=194)
- fail: What colour is the sky on a clear day? Answer in one sentence. (closed=False, correct=False, tokens=1024)

Breakage regressions: 0 at indices []
Kill P1: True, Kill P2: True
