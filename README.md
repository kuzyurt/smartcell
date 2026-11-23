<div align="center">

# smartcell

### still running

**A SO-101 picks a 32 mm part off a conveyor that never stops — and places it in a tote.**

[![The cell, running](media/hero.gif)](media/cell_film.mp4)

`MuJoCo 3.14` · `25 Hz` · `one wrist camera` · `no object state` · `MIT`

[Result](#the-result) · [How it works](#how-it-works) · [The hard part](#the-hard-part) · [Dressed cell](#zero-shot-in-a-dressed-cell) · [Data](#the-data) · [Smooth motion](#smooth-motion) · [Web demo](#web-demo) · [Run it](#run-it) · [Code map](#code-map)

</div>

## The result

Fresh seeds, frozen checkpoint, **200 trials per condition**:

| Condition | Pick | Place |
|---|---:|---:|
| nominal | **99.5%** | **99.0%** |
| tierB — lighting, tint, mount jitter | **99.0%** | **97.5%** |
| boxX — held-out tote corners | **99.5%** | **99.5%** |

A *place* means the part was released, stayed inside the tote, and moved
slower than 0.05 m/s for 0.4 s. Touching the tote does not count.

![Fresh-seed and dressed-scene results](media/results.svg)

The checkpoint and controller were frozen before the final seed stream. The
400 evaluation objects have zero overlap with 2,302 training seeds. Full
protocol, Wilson intervals and caveats: [`docs/validation.md`](docs/validation.md).

## How it works

![Control architecture](media/architecture.svg)

- **A frozen behaviour-cloning policy** reads 8 grayscale wrist frames
  (280 ms of history) plus 12 proprio values, and predicts an 8×6 action
  chunk. Two entries (80 ms) are executed before it looks again.
- **A classical RGB wrist servo** back-projects the part and tote onto the
  known work plane. It leads the moving part, gates the close at predicted
  contact, and fits the tote before descending.
- **A proprioceptive release primitive** confirms a stalled close, a lift, and
  a low descent away from the belt before opening the jaws.

The policy is frozen. The servo adds no learned parameters and no simulator
object state — only pixels, joints, forward kinematics and the commands it
already sent. More in [`docs/design.md`](docs/design.md).

## The hard part

![Training pipeline](media/pipeline.svg)

The first working policy picked about **3%** of the time. The useful history
is the list of things that were actually wrong:

- **The old data described a different machine.** Episodes started at `HOME`,
  the belt stopped at the index station, and "close" was a half-shut claw that
  only looked closed because the servo stalled.
- **The wrist camera clipped the gripper out of its own view.** The near plane
  cut the jaws off. The policy could not see what it was holding.
- **The policy had no sense of its own grip.** Feeding back `SOLL` (commanded
  target) next to `IST` (measured joint) made the gripper tracking error
  observable. A held part and an empty close became different inputs.
- **The belt does not wait for a 200 ms chunk.** 25 Hz, eight-frame history and
  two-entry execution (80 ms) was the difference between chasing and catching.
- **The policy had to survive its own mistakes.** 948 DAgger correction
  episodes, collected from states the policy actually visited, were added to
  training. Validation and test stayed on the original expert split.
- **A grasp is not a place.** The strict metric, a release primitive and a
  tote-fit check turned "the part touched the tote" into "the part stayed
  there".

A lower-validation-loss continuation (`policy_0041f`) failed the rollout gate
and was rejected. Validation loss alone is not a selection metric here.

## Zero-shot in a dressed cell

[![The Blender-dressed cell](media/blender_cell.jpg)](media/policy.mp4)

The same frozen controller, with **no retraining**, in the MuJoCo cell with 49
Blender props imported as **visual-only** geometry (`contype=0`,
`conaffinity=0`). Robot, belt, parts, tote and bin physics are untouched. This
tests visual robustness, not new contact physics.

| Condition | Trials | Pick | Place | Place Wilson 95% |
|---|---:|---:|---:|---|
| nominal | 100 | **100.0%** (100/100) | **99.0%** (99/100) | 94.6–99.8% |
| tierB — lighting, tint, mount jitter | 50 | **100.0%** (50/50) | **100.0%** (50/50) | 92.9–100.0% |
| boxX — held-out tote corners | 50 | **100.0%** (50/50) | **100.0%** (50/50) | 92.9–100.0% |
| nominal, visual servo disabled (control) | 50 | 86.0% (43/50) | 60.0% (30/50) | 46.2–72.4% |

Across all three active dressed conditions, the hybrid controller achieved **199/200 (99.5%)** completed placements. On the same 50 nominal seeds, it placed **49/50 (98.0%)**; with the RGB servo disabled, it placed **30/50 (60.0%)**. The servo is what absorbs the visual shift. Raw reports in [`reports/`](reports/).

## Smooth motion

The arm worked but visibly shook. The cause was not the policy: it emits one
arm target per 40 ms, and the simulator held each target constant before
jumping to the next, so the position servo followed a 25 Hz staircase.
[`smoothing.py`](smoothing.py) interpolates the target inside each step.
The policy, checkpoint and the hash-pinned evaluation files are unchanged;
the validated protocol stays the default.

| Same 8 episodes, actual joint angles at 200 Hz | Original | `smooth` |
|---|---:|---:|
| RMS jerk | 100 % | **43 %** |
| RMS acceleration | 100 % | 55 % |
| high-frequency residual | 100 % | 84 % |

`smooth` placed **44/44** parts (20 nominal, 12 lighting, 12 tote) on a
CPU-only run. That is a small sample next to the 200-trial protocol above and
is not a replacement for it. A stronger variant that also low-pass filtered the
command was calmer but placed only 16/20, because the added lag costs grasps,
so it was dropped. Measurement code: [`tools/measure_smoothing.py`](tools/measure_smoothing.py).

## The data

![Sampled line conditions](media/dataset_variation.png)

| | |
|---|---:|
| training episodes | **1,191** |
| — expert demonstrations | 243 |
| — DAgger corrections | 948 |
| training samples | **549,436** |
| validation / test episodes | 30 / 30 |
| unique training seeds | 2,302 across 42 manifests |
| line speed | 0.068–0.092 m/s |
| part yaw | full ±π |
| tote offset | ±60 mm in x and y |

Every episode samples a new belt speed, part yaw, tote offset, wait pose,
lighting, part tint and camera-mount jitter.

![One expert episode, wrist camera](media/dataset.gif)

[MP4 version](media/dataset.mp4) · raw 128×128 wrist camera at 25 Hz, no
interpolation or colour correction. The dataset is **not shipped**; the code
and the running-line manifest are. See [`docs/dataset.md`](docs/dataset.md).

## Web demo

![Web demo](media/web_demo.png)

A local page with the Blender hall, live policy rollouts and controls for
scene, lighting, tote position, motion mode, visual servo, camera (including a
draggable orbit view), speed and seed. The simulation runs on your machine;
the browser only displays it.

```bash
pip install -r requirements.txt
python web/server.py          # opens http://127.0.0.1:8800
```

With a GPU it runs near real time; on CPU only it is much slower
(`SMARTCELL_ALLOW_CPU=1`). On headless Linux set `MUJOCO_GL=egl`.

## Run it

```bash
pip install -r requirements.txt

# 20 nominal trials in the stock cell
python smartcell/evaluate.py --ckpt ckpt/policy_0041c.pt --n 20

# live viewer: overhead + wrist RGB + the exact tensor the policy sees
python smartcell/watch.py --ckpt ckpt/policy_0041c.pt

# zero-shot in the Blender-dressed cell
python smartcell/evaluate_imported.py --ckpt ckpt/policy_0041c.pt --n 20 --conds nominal

# contract tests
python -m unittest discover -s tests -v
```

## Code map

All Python code lives in [`smartcell/`](smartcell/) as flat modules, run from
the repository root (`python smartcell/evaluate.py ...`). Files listed in
[`docs/provenance.md`](docs/provenance.md) are pinned by SHA-256 and were moved
without content changes, except one documented path fix in `evaluate.py`.

| Role | Files |
|---|---|
| **Run and inspect** | `evaluate.py` (rollouts, conditions, seed streams) · `evaluate_imported.py` (dressed cell) · `watch.py` (live viewer) |
| **Controller** | `model.py` (policy net) · `wrist_vision.py` (RGB servo) · `policy_runtime.py` (dwell, release, placement monitor) · `carry_policy.py` · `current.py` (runtime profile) |
| **Simulation** | `build_cell.py` · `conveyor_env.py` · `conveyor_drive.py` · `arm_ik.py` · `cam_variant.py` · `mjbox.py` |
| **Scene** | `factory_hall.py` · `line_frame.py` · `scene_look.py` · `make_textures.py` · `blender_props.py` |
| **Data** | `expert.py` · `demo_pick.py` · `collect_state.py` · `dagger_collect.py` · `collect_carry_corrections.py` · `render_dataset.py` · `build_cache.py` · `dataset.py` |
| **Smooth motion** | `smoothing.py` (same folder) |
| **Web demo** | `web/` (server, engine, page) · `tools/` (media rendering, smoothness measurement) |
| **Training** | `train.py` · `gpu_guard.py` · `ramcap.py` (memory cap) |

| Folder | Contents |
|---|---|
| `smartcell/` | all Python code (flat modules) and `assets/` (robot/table/conveyor XML, textures, 49 visual-only props) |
| `web/` | local web demo: `server.py`, `engine.py`, `static/` page |
| `tools/` | media rendering and smoothness measurement |
| `ckpt/` | the validated checkpoint `policy_0041c.pt` |
| `docs/` | validation, model card, dataset, design, provenance |
| `reports/` | raw validation and benchmark JSON |
| `media/` | GIFs, videos and figures used here |
| `ds_0011/` | running-line manifest (the dataset itself is not shipped) |
| `tests/` | contract tests for the runtime and the wrist servo |

## Scope

This is a research artifact: **simulation, one 32 mm part, one SO-101, one
conveyor**. It is not a hardware promise, a product, or a safety system. The
visual servo assumes a calibrated camera and a known work plane.

- [Validation and caveats](docs/validation.md)
- [Model card](docs/model-card.md)
- [Dataset](docs/dataset.md)
- [Design notes](docs/design.md)
- [Provenance and hashes](docs/provenance.md)
- [Third-party licences](THIRD_PARTY.md)

MIT licensed. If you use it, [`CITATION.cff`](CITATION.cff) is there.
