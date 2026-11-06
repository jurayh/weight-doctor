# Receipt: blind_goal — Policy ignores the goal channel

> TOY ONLY: 3x3 grid MLP, 20,612 parameters. Not a claim about any larger model.

## Diagnosis

- **blind_goal**: W1 columns for the goal input group have norm 0: the goal channel never reaches layer 1.

Ground truth (catalog): All goal input columns in W1 are zero: the goal pass-through units and goal_present never fire, so neither the reducing moves nor the fallback fire.

## Diff

- Changed: 18 / 20612 weights (0.0873%)
- Edit L2 norm: 4.243, Linf: 1.000
- Broken hash: `a7a403dc2a622170…`
- Edited hash: `520c4736d8ac4f32…`

## Target battery (the failing behaviour)

- Total: 444, passing before edit: 0, failing after edit: 0

## Breakage battery (everything else)

- Total: 60, failing after edit: 0

## Verdict: FIXED
