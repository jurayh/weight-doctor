# Receipt: blind_hazard — Policy ignores the hazard channel

> TOY ONLY: 3x3 grid MLP, 20,612 parameters. Not a claim about any larger model.

## Diagnosis

- **blind_hazard**: W1 columns for the hazard input group have norm 0: the hazard channel never reaches layer 1.

Ground truth (catalog): All hazard input columns in W1 are zero: the hazard pass-through units never fire, so the hazard term in the safe_reducing units is missing and the policy steps onto hazards on its greedy path.

## Diff

- Changed: 9 / 20612 weights (0.0437%)
- Edit L2 norm: 3.000, Linf: 1.000
- Broken hash: `69e2f2cf268c7eed…`
- Edited hash: `520c4736d8ac4f32…`

## Target battery (the failing behaviour)

- Total: 72, passing before edit: 0, failing after edit: 0

## Breakage battery (everything else)

- Total: 432, failing after edit: 0

## Verdict: FIXED
