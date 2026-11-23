"""Blender set dressing as visual-only MuJoCo geoms.

The OBJ files come from _export_mj_props.py: one mesh per PROPS object, world
coordinates (Blender Z-up == MuJoCo Z-up), flat linear RGBA sampled from the
Blender material.  Everything here is contype=0/conaffinity=0: it can never
touch the physics of the cell (robot, belt, cubes, tote, bin).
"""
import json
import os

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "assets", "blender_props")


def _name(s):
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in s)


def add_props(spec, manifest=None):
    if manifest is None:
        with open(os.path.join(DIR, "manifest.json")) as fh:
            manifest = json.load(fh)
    W = spec.worldbody
    n = 0
    for e in manifest:
        mesh = spec.add_mesh()
        mesh.name = "blp_" + _name(e["mesh"])
        mesh.file = os.path.join(DIR, e["file"]).replace("\\", "/")
        g = W.add_geom()
        g.name = "bprop_" + _name(e["name"])
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh.name
        g.contype = 0
        g.conaffinity = 0
        g.rgba = list(e["rgba"])
        g.group = 1
        n += 1
    return n
