# Design notes

![Control architecture](../media/architecture.svg)

## The shape of the system

It is a hybrid, not an end-to-end retrain.

1. **A frozen behaviour-cloning policy** reads eight grayscale wrist frames
   (280 ms of history) and 12 proprio values. It predicts an 8×6 action chunk;
   the evaluator executes two entries (80 ms) before looking again.
2. **A classical RGB wrist servo** runs on the same camera stream. It
   back-projects pixels onto the known work plane, localises the part top and
   the tote edge, and nudges the arm/grip targets during the last centimetres.
3. **A proprioceptive release primitive** confirms a stalled close, a lift, and
   a low descent away from the belt before opening the jaws.

The policy is frozen. The servo adds no learned parameters and no simulator
object state. The only inputs are RGB pixels, joint angles, forward kinematics,
and the actuator commands.

## The 3% start

The first policy was not a small improvement away from working. It picked about
**3%** of the time. The useful history is the list of things that were actually
wrong, not a single training run:

- **The old data described a different world.** Early datasets started at
  `HOME`, stopped the belt at the index station, and commanded a half-shut claw
  (`0.02 rad`) that only looked closed because the servo stalled. The new
  contract starts at the wait pose, never stops the line, and commands the
  actuator's real lower bound.
- **The wrist camera clipped the gripper out of its own view.** The renderer
  near plane cut the jaws off. The policy could not see what it was holding.
- **The policy had no sense of its own grip.** Feeding back `SOLL` (the command
  actually sent) alongside `IST` (the measured joint) made the gripper tracking
  error observable. A held part and an empty close became different inputs.
- **The belt does not wait for a 200 ms open-loop chunk.** Moving to 25 Hz with
  eight-frame history and two-entry execution (80 ms) was the difference
  between chasing and catching.
- **The policy had to survive its own mistakes.** 948 DAgger correction
  episodes, collected from states the policy actually visited, were added to
  the training set. Validation and test stayed on the original expert split.
- **A grasp is not a place.** The strict metric requires release, retention in
  the tote, and ≤0.05 m/s for 0.4 s. A release primitive and a tote-fit check
  turned "the part touched the tote" into "the part stayed there".
- **The RGB servo fixed the last centimetre.** The base policy still handles
  the wait and the approach. The servo tracks the moving part, leads it by
  0.25 s, gates the close at predicted contact, and fits the tote silhouette
  before descending.
- **One render-timing bug mattered.** MuJoCo `implicitfast` renders kinematics
  before the final hinge integration. The first servo version paired the image
  with post-step joints; v9 compensates with `q − dt·v` and a 5 ms offset. The
  independent geometry test agrees with the rendered camera pose to `1e-12`.

## What was rejected

- A continuation checkpoint (`policy_0041f`) had lower validation loss but
  catastrophic rollout performance. It was not promoted. Validation loss alone
  is not a selection metric for this task.
- Temporal ensembling, entry-edge dwell, carry lookahead, and a learned carry
  head were all tested. They were neutral or worse and stay off by default.
- The visual servo is deliberately not a second neural network. The task has a
  known planar geometry and a calibrated camera; using it is cheaper and easier
  to audit than another learned head.

## Why the result is believable

The controller and checkpoint were frozen before the final seed stream. The
fresh-seed audit checks that the 400 evaluation objects do not overlap 2,302
training seeds or 280 historical evaluation seeds. The final report includes
source hashes, runtime settings, per-episode records, and Wilson intervals.
The 17 runtime/vision tests and a 12-trial default regression are included.

The claim is narrow on purpose: **simulation, one 32 mm part, one SO-101, one
conveyor**. It is a research artifact, not a hardware promise.
