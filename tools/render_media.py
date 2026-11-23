"""Render the web page media: hero film, before/after clips, poster, metrics.

    python tools/render_media.py --what all

Runs the frozen policy in the Blender-dressed cell on fixed seeds and records
frames at the 25 Hz policy grid. ``before`` uses the validated command path,
``after`` the interpolated one (``smoothing.py``). Needs a working MuJoCo GL
backend (MUJOCO_GL=egl|osmesa on headless Linux).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "smartcell"))

import numpy as np
import mujoco
from PIL import Image

from web.engine import Engine, EV, Cell, V
import smoothing

OUT = os.path.join(ROOT, "media")


class Writer:
    def __init__(self, path, w, h, fps=25):
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
             "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def record(eng, cell, profile, seed, path, size, cam, orbit_fn=None, skip_wait=True,
           poster=None):
    smoothing.install(profile)
    EV.VISUAL_PICK = EV.VISUAL_PLACE = True
    rend = mujoco.Renderer(cell.r.model, height=size[1], width=size[0])
    rend._scene_option.geomgroup[:] = [1] * 6
    wr = Writer(path, *size)
    free = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(cell.r.model, free)
    state = dict(n=0, saved=False)
    arms = []

    def on_step(s):
        ph = s["phase"]
        if ph == "policy":
            arms.append(np.asarray(s["arm"], float))
        if ph == "wait" and skip_wait and state["n"] % 4:
            state["n"] += 1
            return
        t = state["n"]
        state["n"] += 1
        if orbit_fn:
            orbit_fn(free, t)
            rend.update_scene(cell.r.data, camera=free)
        else:
            rend.update_scene(cell.r.data, camera=cell.cam_id[cam])
        img = rend.render()
        wr.write(img)
        if poster and not state["saved"] and ph == "policy" and s["t"] > 1.0:
            Image.fromarray(img).save(poster, quality=90); state["saved"] = True

    rng = np.random.default_rng(seed)
    picked, placed = EV.rollout(cell.env, eng.model, cell.mount, cell.rend_ev, cell.visap,
                                None, rng, eng.stack, None,
                                lrng=np.random.default_rng(777), on_step=on_step)
    wr.close()
    rend.close()
    return picked, placed, np.array(arms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["all", "hero", "compare"])
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()
    eng = Engine()
    cell = Cell("blender")
    prof = "smooth"

    if a.what in ("all", "compare"):
        def close_cam(c, t):
            c.lookat[:] = (0.0, -0.12, 0.74); c.distance = 1.05
            c.azimuth, c.elevation = 128.0, -22.0
        for name, p in (("before", "off"), ("after", prof)):
            ok = record(eng, cell, p, a.seed, os.path.join(OUT, f"{name}.mp4"),
                        (640, 480), None, close_cam, skip_wait=True)
            print(name, "pick/place", ok[:2], flush=True)

    if a.what in ("all", "hero"):
        def drift(c, t):
            k = t / 600.0
            c.lookat[:] = (0.0, -0.3, 0.7)
            c.azimuth = 118.0 + 34.0 * np.sin(k * np.pi)
            c.elevation = -21.0 + 4.0 * np.cos(k * np.pi)
            c.distance = 2.5 - 0.5 * k
        ok = record(eng, cell, prof, a.seed + 1, os.path.join(OUT, "hero.mp4"),
                    (1280, 720), None, drift, skip_wait=True,
                    poster=os.path.join(OUT, "hero_poster.jpg"))
        print("hero pick/place", ok[:2])


if __name__ == "__main__":
    main()
