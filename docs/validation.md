# Validation

## The claim

With the checkpoint and controller frozen, the cell picks and places the part
in **more than 95%** of trials in every existing MuJoCo condition:

- **nominal** — stock cell and stock camera mount;
- **tierB** — fresh per-episode lighting, part tint/specular and mount jitter;
- **boxX** — held-out tote positions at three corners of the placement region.

The metric is strict: the part must be released, inside the tote, and slower
than 0.05 m/s for 0.4 s. Merely touching the tote does not count.

## Frozen before the final stream

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
found **zero overlap** with 2,302 training seeds, 280 historical evaluation
seeds, or 123 historical reports across 42 manifests.

| Condition | Picks / 200 | Pick rate | Pick Wilson 95% | Places / 200 | Place rate | Place Wilson 95% | Median place |
|---|---:|---:|---|---:|---:|---|---:|
| nominal | 199 | **99.5%** | 97.2–99.9% | 198 | **99.0%** | 96.4–99.7% | 13.52 s |
| tierB | 198 | **99.0%** | 96.4–99.7% | 195 | **97.5%** | 94.3–98.9% | 13.68 s |
| boxX | 199 | **99.5%** | 97.2–99.9% | 199 | **99.5%** | 97.2–99.9% | 13.60 s |

Source: `reports/visual_both_v12_final_validation_20261026.json`.

### Caveats

- **Nominal and boxX share object seeds.** boxX changes the tote position, not
  the part stream. The aggregate 600 trials are therefore not 600 independent
  objects; read the per-condition rows.
- **tierB is the weakest condition.** 97.5% place has a Wilson lower bound of
  94.3%, just below 95%. The point estimate clears the bar; the interval does
  not. This is stated rather than rounded away.
- **The stock mount ignores legacy jitter.** Back/tilt/azimuth displacement of
  the physical camera was not part of this test.
- **Simulation only.** These are measured MuJoCo rates, not a hardware
  guarantee.

## Matched A/B: what the wrist servo actually bought

The fresh-seed run measures the final system. To isolate the contribution of
the RGB wrist servo, 100 objects per condition were replayed through the same
base policy with the servo off and on.

| Condition | Baseline picks | Baseline places | Servo picks | Servo places | Recovered | Regressed | Paired exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | 91 | 81 | 99 | **99** | +19 | 1 | 4.0 × 10⁻⁵ |
| tierB | 90 | 76 | 98 | **96** | +23 | 3 | 8.8 × 10⁻⁵ |
| boxX | 91 | 75 | 99 | **99** | +24 | 0 | 1.2 × 10⁻⁷ |

Source: `reports/visual_both_v12_matched_comparison_20261026.json`.

> **Footnote on the old 77% → 98% number.** An earlier internal summary
> aggregated the three matched baseline place counts (81 + 76 + 75 = 232/300,
> 77.3%) and the three servo counts (99 + 96 + 99 = 294/300, 98.0%). Those
> aggregates mix three different conditions and hide the per-condition spread.
> The table above is the correct way to read the A/B. The headline claim in
> this repository is the fresh-seed result, not that aggregate.

## Zero-shot in the Blender-dressed scene

The same frozen checkpoint and default hybrid controller were run in the
MuJoCo cell with the Blender set dressing imported as **visual-only** geoms:
49 OBJ props, `contype=0`, `conaffinity=0`. Robot, belt, parts, tote and bin
are unchanged. No retraining, fine-tuning or threshold changes.

| Condition | Trials | Picks | Pick rate | Pick Wilson 95% | Places | Place rate | Place Wilson 95% | Median place |
|---|---:|---:|---:|---|---:|---:|---|---:|
| nominal, dressed | 100 | 100 | **100.0%** | 96.3–100.0% | 99 | **99.0%** | 94.6–99.8% | 13.56 s |
| tierB, dressed | 50 | 50 | **100.0%** | 92.9–100.0% | 50 | **100.0%** | 92.9–100.0% | 13.64 s |
| boxX, dressed | 50 | 50 | **100.0%** | 92.9–100.0% | 50 | **100.0%** | 92.9–100.0% | 13.52 s |
| nominal, dressed, visual off | 50 | 43 | 86.0% | 73.8–93.0% | 30 | 60.0% | 46.2–72.4% | 13.80 s |

Seed base `31102601`, fresh stream. Source reports in `reports/blender_zero_shot_20261031_*.json`.

### Analysis

- **200/200 picks (100.0%) and 199/200 places (99.5%)** across the three active conditions.
- **The only miss** was one nominal episode (seed 1798426699). The part was picked at $t=9.56\text{ s}$ and first entered the tote at $t=19.68\text{ s}$; the 20.0 s horizon ended 0.32 s later, before the 0.4 s settle window. The report records no primitive release and the part was not held at the end, so this is a horizon/settling miss, not a failed grasp.
- **Control ablation:** On the same 50 nominal seeds, placement is 49/50 (98.0%) with the RGB wrist servo and 30/50 (60.0%) without it. Of the 20 failures, 13 were picked but not placed before the horizon and 7 were never picked. This isolates the servo's contribution to the visual-shift robustness.

## Runtime verification

- **17 runtime/vision tests pass** from the clean tree:
  `python -m unittest discover -s tests -v`.
- **12-trial default regression** reproduces the expected outcomes and
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
