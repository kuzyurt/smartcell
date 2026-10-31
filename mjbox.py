"""Shared geometry helpers, used by both build_cell.py and factory_hall.py."""
import mujoco
import numpy as np


def col(v):
    """Column vector for MjSpec numeric fields."""
    return np.asarray(v, dtype=float).reshape(-1, 1)


def box(parent, name, xr, yr, zr, rgba=(0.45, 0.47, 0.52, 1.0),
        friction=None, mass=None, contype=1, conaffinity=1):
    """Axis-aligned box from opposite corners, so joints butt instead of overlap."""
    g = parent.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = col([(xr[1] - xr[0]) / 2, (yr[1] - yr[0]) / 2, (zr[1] - zr[0]) / 2])
    g.pos = col([(xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2])
    g.rgba = col(list(rgba))
    g.contype = contype
    g.conaffinity = conaffinity
    if friction is not None:
        g.friction = col(friction)
    if mass is not None:
        g.mass = mass
    return g
