# Receipt — C1_subgroup_ablation
Family: L15 column ablation (best subgroup)
Scope: Qwen3.5-0.8B, thinking mode ON, Kaggle T4, 1024-token cap, this battery only. No claim about other models, modes, or batteries.
Diff: 114688 weights changed (0.00015243097203410542), L2=3.0371, layer-limited=True, edit_type=weight edit
Target: 5/9
- PASS: Maria has 24 apples. She gives 7 to her brother and buys 15 more. How  (closed=True, correct=True, tokens=699)
- fail: A baker makes 5 trays with 12 muffins each. He sells 38 muffins. How m (closed=False, correct=False, tokens=1024)
- PASS: Tom reads 18 pages a day. How many pages does he read in 2 weeks? (closed=True, correct=True, tokens=779)
- fail: A rectangle is 9 cm long and 4 cm wide. What is its area in square cm? (closed=False, correct=False, tokens=1024)
- fail: A train travels 60 km per hour for 3 hours, then travels 40 km more. H (closed=False, correct=False, tokens=1024)
- PASS: Jake earns $8 per hour and works 6 hours a day, 5 days a week. How muc (closed=True, correct=True, tokens=773)
- PASS: A store sells notebooks for $3 each. Ana buys 7 notebooks and pays wit (closed=True, correct=True, tokens=693)
- PASS: What is the capital of France? Answer in one sentence. (closed=True, correct=True, tokens=246)
- fail: What colour is the sky on a clear day? Answer in one sentence. (closed=False, correct=False, tokens=1024)

Breakage regressions: 2 at indices [0, 10]
Kill P1: False, Kill P2: True
