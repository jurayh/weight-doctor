# Receipt — C2_group_ablation
Family: L15 column ablation (best group)
Scope: Qwen3.5-0.8B, thinking mode ON, Kaggle T4, 1024-token cap, this battery only. No claim about other models, modes, or batteries.
Diff: 458752 weights changed (0.0006097238881364217), L2=5.957, layer-limited=True, edit_type=weight edit
Target: 6/9
- PASS: Maria has 24 apples. She gives 7 to her brother and buys 15 more. How  (closed=True, correct=True, tokens=720)
- fail: A baker makes 5 trays with 12 muffins each. He sells 38 muffins. How m (closed=False, correct=False, tokens=1024)
- PASS: Tom reads 18 pages a day. How many pages does he read in 2 weeks? (closed=True, correct=True, tokens=914)
- PASS: A rectangle is 9 cm long and 4 cm wide. What is its area in square cm? (closed=True, correct=True, tokens=542)
- fail: A train travels 60 km per hour for 3 hours, then travels 40 km more. H (closed=False, correct=False, tokens=1024)
- PASS: Jake earns $8 per hour and works 6 hours a day, 5 days a week. How muc (closed=True, correct=True, tokens=848)
- fail: A store sells notebooks for $3 each. Ana buys 7 notebooks and pays wit (closed=False, correct=False, tokens=1024)
- PASS: What is the capital of France? Answer in one sentence. (closed=True, correct=True, tokens=190)
- PASS: What colour is the sky on a clear day? Answer in one sentence. (closed=True, correct=True, tokens=240)

Breakage regressions: 3 at indices [0, 1, 12]
Kill P1: False, Kill P2: True
