"""evaluate.py on the cell with the Blender set dressing imported as visuals.

Physics is untouched: every imported geom is contype=0/conaffinity=0, and the
robot, cubes, belt, tote and bin still come from build_cell.

Run: python evaluate_imported.py --ckpt ckpt/policy_0041c.pt --n 5 --conds nominal
"""
import runpy
import sys

import build_cell as C
import blender_props

N_ADDED = blender_props.add_props


def build_model():
    spec = C.build_spec()
    n = N_ADDED(spec)
    model = spec.compile()
    model = C.scene_look.polish(model)
    print(f"blender set dressing: {n} visual geoms, physics untouched",
          flush=True)
    return model


C.build_model = build_model

runpy.run_path("evaluate.py", run_name="__main__")
