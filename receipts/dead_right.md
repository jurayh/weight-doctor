# Receipt: dead_right — Policy has a dead RIGHT action

> TOY ONLY: 3x3 grid MLP, 20,612 parameters. Not a claim about any larger model.

## Diagnosis

- **dead_action_right**: Output row for RIGHT is zero and its bias is -5, so RIGHT can never win argmax.

Ground truth (catalog): The RIGHT output row is zeroed and its bias is -5, so RIGHT can never win argmax; any goal east of the start is unreachable.

## Diff

- Changed: 13 / 20612 weights (0.0631%)
- Edit L2 norm: 7.416, Linf: 5.000
- Broken hash: `29a3751ae0a47e5f…`
- Edited hash: `520c4736d8ac4f32…`

## Target battery (the failing behaviour)

- Total: 195, passing before edit: 0, failing after edit: 0

## Breakage battery (everything else)

- Total: 309, failing after edit: 0

## Verdict: FIXED
