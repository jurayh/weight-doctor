# Receipt: stuck_center — Policy is stuck in a local minimum at the centre

> TOY ONLY: 3x3 grid MLP, 20,612 parameters. Not a claim about any larger model.

## Diagnosis

- **spurious_weight**: Wout entries above the schema maximum 2: a spurious connection dominates its action wherever its hidden unit fires.

Ground truth (catalog): One spurious Wout weight (centre fallback-UP, 1 -> 11) makes UP dominate at the centre whatever the goal is, so paths through the centre with a southern goal oscillate centre -> north -> centre and loop.

## Diff

- Changed: 1 / 20612 weights (0.0049%)
- Edit L2 norm: 10.000, Linf: 10.000
- Broken hash: `5a60b891827fa4df…`
- Edited hash: `520c4736d8ac4f32…`

## Target battery (the failing behaviour)

- Total: 102, passing before edit: 0, failing after edit: 0

## Breakage battery (everything else)

- Total: 402, failing after edit: 0

## Verdict: FIXED
