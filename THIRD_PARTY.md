# Third-party assets and dependencies

The MIT licence in `LICENSE` covers the original code and media in this
repository. The following third-party assets keep their own licences.

## Vendored model assets

| Asset | Source | Licence | What was changed |
|---|---|---|---|
| SO-101 robot | [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie) `robotstudio_so101` | Apache-2.0 | unmodified, loaded from the local menagerie checkout |
| Conveyor section | [`google-deepmind/mujoco`](https://github.com/google-deepmind/mujoco) `model/surfacevel/carousel.xml` | Apache-2.0 | cross-section adapted, tail lengthened 200 mm; upstream copy kept at `smartcell/assets/_upstream_mujoco_carousel.xml` |
| Work table | [`vikashplus/furniture_sim`](https://github.com/vikashplus/furniture_sim) `simpleTable` | Apache-2.0 | joints fixed; upstream copy kept at `smartcell/assets/_upstream_simpleTable.xml`; licence text at `smartcell/assets/LICENSE_furniture_sim.txt` |

The factory hall, conveyor line frame, textures and lighting rig are original
work built in `factory_hall.py`, `line_frame.py` and `scene_look.py`.

## Blender set dressing

`smartcell/assets/blender_props/` contains 49 OBJ meshes exported from
[Poly Haven](https://polyhaven.com/) models. Poly Haven assets are **CC0**.
Only geometry and a flat linear RGBA colour sampled from the source material
are included; no Poly Haven textures are redistributed. The meshes are loaded
as visual-only geoms (`contype=0`, `conaffinity=0`) and never affect physics.
See `smartcell/assets/blender_props/PROVENANCE.md`.

## Python dependencies

| Package | Licence |
|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv) | Apache-2.0 |
| [Pillow](https://github.com/python-pillow/Pillow) | HPND |

Dependency licences apply to those packages, not to this repository.
