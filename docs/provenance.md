# Provenance

This repository is a curated export of a private research tree, not a
reconstruction from memory. The checkpoint, controller and reports below are
the exact artifacts that produced the published numbers.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `ckpt/policy_0041c.pt` | `9d5430c20649bc3029e896b2d96976367c9dd0cc7faa4629c73ec91e24b87dad` |
| `wrist_vision.py` (RGB servo) | `0a0296eb42a999b2662592d3980ea0833a8d7aeb568104b81dc13083b052c4b6` |
| `evaluate.py` | `b1f241ab682c47094870bb24f1fbf11417ef06de34c657f11564eb53aaa3bd8a` |
| `current.py` (runtime profile) | `816f5c6438b3ec8c16e1a817fc5dcf3bb1ec50f7c063b1b4bd8178929e925f1b` |
| `model.py` | `9b4f2f4eb60bf78295c725fbaa1831ee155e168bbc389cfe6772f61198bc0a97` |
| `policy_runtime.py` | `edcd03134a79bea69d94cdbb3420721463703cbd7998fe928ad78a4b32c4c2ee` |
| `build_cell.py` | `71cb07cbb05cdac9839bb03983cec3cd58de06b4152d7c63522f763c746e89fb` |
| `conveyor_env.py` | `16ad77b01bd411e36dcc3eda2d14e41367e3cc63f66695ec28630124f45b3ab3` |

The controller hash and checkpoint hash match the values recorded in
`reports/visual_both_v12_final_validation_20261026.json`.

## Environment used for the published runs

| | |
|---|---|
| OS | Windows 11 |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |
| MuJoCo | 3.14.0 |
| GPU | NVIDIA GeForce RTX 2060, 6 GB |
| RAM cap | 5 GB (hard, enforced by `ramcap.py`) |
| Render backend | WGL/GLFW (Windows default) |

## Data lineage

The validated checkpoint was trained on:

- `ds_0011` + `cache_0011` — 243 expert demonstrations, 25 Hz, running line;
- 12 DAgger rounds (`ds_0011_d16` … `ds_0011_d27`) — 692 correction episodes,
  train split only;
- two visual-domain rounds (`ds_0016_look`, `ds_0017_look`) — 256 additional
  correction episodes, for 948 corrections in total.

Final split: **1,191 train / 30 validation / 30 test episodes**, 549,436 train
samples. The dataset itself is not shipped; `ds_0011/manifest.jsonl` is kept
so `evaluate.py --ds ds_0011` can read the running-line configuration.

The 400 objects in the final fresh-seed validation are disjoint from 2,302
training seeds and 280 historical evaluation seeds. The audit is
`reports/visual_both_v12_fresh_seed_audit_20261026.json`.

## Asset provenance

- Robot, conveyor and table: Apache-2.0, see [`THIRD_PARTY.md`](../THIRD_PARTY.md)
  and the long-form notes in [`assets/PROVENANCE.md`](../assets/PROVENANCE.md).
- Blender set dressing: Poly Haven CC0, geometry only, see
  [`assets/blender_props/PROVENANCE.md`](../assets/blender_props/PROVENANCE.md).
- Hall, line frame, textures and lighting: original work in this repository.

## Report map

| Report | What it contains |
|---|---|
| `reports/visual_both_v12_final_validation_20261026.json` | 200 fresh-seed trials per condition, Wilson intervals |
| `reports/visual_both_v12_fresh_seed_audit_20261026.json` | seed-overlap audit |
| `reports/visual_both_v12_matched_comparison_20261026.json` | matched 100-object A/B |
| `reports/visual_default_regression_20261026.json` | 12-trial default-profile regression |
| `reports/blender_zero_shot_20261031_*.json` | zero-shot benchmark in the dressed scene |

The development tree had no Git history. This repository is the export; the
hashes above are the authority, not a commit message.
