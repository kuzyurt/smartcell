# Validation

## Claim

With the checkpoint and controller frozen, the cell picks and places the part
in more than 95% of trials in each of the three MuJoCo test conditions:

- `nominal`: the default cell and camera mount.
- `tierB`: lighting, part colour and specularity, and camera-mount offset are
  drawn fresh for every episode.
- `boxX`: the box is placed at three corners of the placement region that lie
  outside the training range.

A trial counts as placed only if the part was released, is inside the box and
moved slower than 0.05 m/s for 0.4 s. Touching the box does not count.

## Frozen setup

| | |
|---|---|
| checkpoint | `ckpt/policy_0041c.pt` |
| checkpoint SHA-256 | `9d5430c2…b87dad` |
| controller | `wrist_vision.py` v12 |
| controller SHA-256 | `0a0296eb…c4b6` |
| runtime | `current.py` default profile |
| seed base | `27102601` |
| trials | 200 per condition |

Runtime profile: `exec=2`, decision-edge dwell `0.4 s`, place settle `0.4 s`,
release primitive on, visual pick/place on, visual close `0.4 s`, takeover
height `0.81 m`, grip threshold `−0.4`, grip snap on. Full hashes are in
[`provenance.md`](provenance.md).

## Fresh seeds

The 400 object seeds were generated after the controller was frozen. The audit
found zero overlap with 2,302 training seeds, 280 historical evaluation
seeds, or 123 historical reports across 42 manifests.

| Condition | Picks / 200 | Pick rate | Pick Wilson 95% | Places / 200 | Place rate | Place Wilson 95% | Median place |
|---|---:|---:|---|---:|---:|---|---:|
| nominal | 199 | 99.5% | 97.2–99.9% | 198 | 99.0% | 96.4–99.7% | 13.52 s |
| tierB | 198 | 99.0% | 96.4–99.7% | 195 | 97.5% | 94.3–98.9% | 13.68 s |
| boxX | 199 | 99.5% | 97.2–99.9% | 199 | 99.5% | 97.2–99.9% | 13.60 s |

Source: `reports/visual_both_v12_final_validation_20261026.json`.

### Limitations

- `nominal` and `boxX` use the same object seeds; `boxX` only moves the box.
  The 600 trials together are therefore fewer than 600 independent objects,
  and the rows should be read separately.
- `tierB` is the weakest condition. Its place rate of 97.5% has a Wilson lower
  bound of 94.3%, slightly below the 95% target, even though the point
  estimate is above it.
- The default camera mount does not include the older back, tilt and azimuth
  offsets of the physical camera. Those were not tested here.
- These are rates measured in MuJoCo. They do not carry over to hardware
  automatically.

## Matched comparison with and without the wrist servo

The fresh-seed run measures the final system. To isolate the contribution of
the RGB wrist servo, 100 objects per condition were replayed through the same
base policy with the servo off and on.

| Condition | Baseline picks | Baseline places | Servo picks | Servo places | Recovered | Regressed | Paired exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | 91 | 81 | 99 | 99 | +19 | 1 | 4.0 × 10⁻⁵ |
| tierB | 90 | 76 | 98 | 96 | +23 | 3 | 8.8 × 10⁻⁵ |
| boxX | 91 | 75 | 99 | 99 | +24 | 0 | 1.2 × 10⁻⁷ |

Source: `reports/visual_both_v12_matched_comparison_20261026.json`.

An earlier summary quoted this comparison as 77% → 98%. That figure adds up
the three baseline place counts (81 + 76 + 75 = 232/300) and the three servo
counts (99 + 96 + 99 = 294/300) across different conditions, which hides the
spread between them. The per-condition table above is the better way to read
it, and the main result of this repository is the fresh-seed run.

## Furnished hall (Blender props)

The same frozen checkpoint and default hybrid controller were run in the
MuJoCo cell with the Blender set dressing imported as visual-only geoms:
49 OBJ props, `contype=0`, `conaffinity=0`. Robot, belt, parts, tote and bin
are unchanged. No retraining, fine-tuning or threshold changes.

| Condition | Trials | Picks | Pick rate | Pick Wilson 95% | Places | Place rate | Place Wilson 95% | Median place |
|---|---:|---:|---:|---|---:|---:|---|---:|
| nominal, dressed | 100 | 100 | 100.0% | 96.3–100.0% | 99 | 99.0% | 94.6–99.8% | 13.56 s |
| tierB, dressed | 50 | 50 | 100.0% | 92.9–100.0% | 50 | 100.0% | 92.9–100.0% | 13.62 s |
| boxX, dressed | 50 | 50 | 100.0% | 92.9–100.0% | 50 | 100.0% | 92.9–100.0% | 13.48 s |
| nominal, dressed, visual off | 50 | 43 | 86.0% | 73.8–93.0% | 30 | 60.0% | 46.2–72.4% | 13.78 s |

Seed base `31102601`, fresh stream. Source reports in `reports/blender_zero_shot_20261031_*.json`.

### Notes

- 200 of 200 parts were picked and 199 of 200 placed across the three
  conditions with the servo on.
- The one miss was a `nominal` episode (seed 1798426699). The part was picked
  at 9.56 s and first entered the box at 19.68 s. The 20.0 s time limit ended
  0.32 s later, before the 0.4 s settle window was complete. The part was not
  held at the end and no release was recorded, so the episode ran out of time
  after a successful grasp.
- On the same 50 `nominal` seeds, 49 parts (98.0%) were placed with the RGB
  servo and 30 (60.0%) without it. Of the 20 failures without the servo, 13
  were picked but not placed in time and 7 were never picked.

## Runtime verification

- 17 runtime/vision tests pass from the clean tree:
  `python -m unittest discover -s tests -v`.
- 12-trial default regression reproduces the expected outcomes and
  timings: `reports/visual_default_regression_20261026.json`.
- The viewer was checked against 160 policy stacks; the camera-pose geometry
  test agrees with the rendered pose to `1e-12`.

## Reproduce

```bash
pip install -r requirements.txt

python smartcell/evaluate.py --ckpt ckpt/policy_0041c.pt \
  --n 200 --conds nominal --seed-base 27102601 \
  --out reports/repro_nominal.json

python smartcell/evaluate.py --ckpt ckpt/policy_0041c.pt \
  --n 200 --conds tierB --seed-base 27102601 \
  --out reports/repro_tierB.json

python smartcell/evaluate.py --ckpt ckpt/policy_0041c.pt \
  --n 200 --conds boxX --seed-base 27102601 \
  --out reports/repro_boxX.json

# dressed-scene zero-shot
python smartcell/evaluate_imported.py --ckpt ckpt/policy_0041c.pt \
  --n 100 --conds nominal --seed-base 31102601 \
  --out reports/repro_dressed_nominal.json
```

The dataset is not shipped. `ds_0011/manifest.jsonl` supplies the running-line
configuration; the rollouts themselves are generated by the simulator.
