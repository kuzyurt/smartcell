# Model card

## What it is

A 0.979 M-parameter behaviour-cloning policy for a SO-101 arm on a running
conveyor. It is one half of a hybrid controller: the policy produces nominal
motion, and a classical RGB wrist servo plus a proprioceptive release
primitive correct the final centimetres.

This is a research project in simulation. It has not been tested on hardware
and is not suitable for production or safety-relevant use.

## Architecture

| | |
|---|---|
| Vision tower | 4 × conv (48 → 72 → 128 → 192), 3×3, stride 2, BN + ReLU, global average pool |
| Proprio tower | MLP 12 → 128 → 128 |
| Fusion | concat 320 → FC 256 → 8 × 6 |
| Parameters | 0.979 M |
| Output activation | tanh |

## Inputs

| Input | Shape | Details |
|---|---|---|
| wrist frames | `8 × 96 × 96` | 8 grayscale frames stacked as channels, 40 ms apart, 280 ms lookback |
| proprio | `12` | `IST 6` measured joints + `SOLL 6` commanded targets |

`SOLL − IST` is the tracking error. On the gripper it is a direct held-object
signal: a closed jaw stalled on a part looks different from an empty close.
No simulator object pose, cube state or tote state enters the policy.

## Output

An `8 × 6` action chunk at 25 Hz: five arm targets and one gripper target,
normalised to `[-1, 1]`. The evaluator executes 2 entries (80 ms) before
re-observing.

## Training

- Behaviour cloning with a Huber loss (`delta=1`) on all 48 chunk entries.
- AdamW, cosine schedule, AMP, batch 256.
- 8-frame history, 8-step chunk.
- Manifold augmentation: ±2° wrist rotation, ±0.015 rad IST/SOLL noise,
  gripper-gap scaling, proprio dropout.
- Checkpoint selection uses validation loss, but promotion is gated on
  closed-loop rollout performance. A lower-validation-loss continuation
  (`policy_0041f`) failed the rollout gate and was rejected.

Training data: 1,191 episodes / 549,436 samples (243 expert + 948 DAgger
corrections). Validation and test are 30 expert episodes each and never
include DAgger data. See [`dataset.md`](dataset.md).

## Intended use

- Reproducing the reported simulation result.
- Research on behaviour cloning, visual servoing and conveyor manipulation.
- A baseline for sim-to-real work on small pick-and-place cells.

## Out of scope

- Real hardware deployment without independent safety review.
- Arbitrary objects, grippers, cameras or conveyor layouts.
- Safety-critical operation, human proximity, or unattended running.
- Claims that the reported simulation rates transfer to a physical cell.

## Evaluation

The headline result is 200 fresh-seed trials per condition with the checkpoint
and controller frozen before the seed stream:

| Condition | Pick | Place |
|---|---:|---:|
| nominal | 99.5% | 99.0% |
| tierB (visual shift) | 99.0% | 97.5% |
| boxX (held-out tote corners) | 99.5% | 99.5% |

A separate zero-shot run places the same frozen controller in a
Blender-dressed scene with 49 visual-only props. Protocol, intervals and
caveats are in [`validation.md`](validation.md).

## Limitations

- One 32 mm part, one SO-101, one conveyor geometry, simulation only.
- The visual servo assumes a calibrated camera and a known work plane.
- Nominal and boxX share object seeds, so the aggregate 600 is not 600
  independent objects.
- tierB place is 97.5% with a Wilson 95% lower bound of 94.3%, just under 95%.
- The stock camera mount ignores the legacy back/tilt/azimuth jitter; physical
  camera displacement was not tested.
- The Blender props are visual-only. That benchmark tests visual robustness,
  not new contact physics.

## Safety and ethics

The controller has no obstacle avoidance, no force limiting beyond the
actuator limits and no detection of people. It should not be run on hardware
near people or left running unattended.
