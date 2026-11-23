"""Simulation engine for the smartcell web demo.

Runs the *unmodified* ``evaluate.rollout`` with an ``on_step`` observer that
renders frames for the browser. MuJoCo's GL context is bound to the thread
that created the renderers, so ``Engine.run_forever`` must be called on the
thread that built the Engine (the HTTP server lives on daemon threads).
"""
import io
import json
import os
import sys
import threading
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "smartcell"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                       # the project uses repo-relative paths

import numpy as np
import mujoco
import torch
from PIL import Image

import build_cell as C
import blender_props
import cam_variant as V
import collect_state as CS
import current
import evaluate as EV
import gpu_guard
import smoothing
from collect_state import RandEnv
from conveyor_env import ConveyorPickEnv
from model import Policy, load_state
from render_dataset import sample_vis, VisApply

SCENES = ("blender", "stock")
CONDITIONS = ("nominal", "tierB", "boxX")
CAMERAS = ("hall", "preview", "overhead", "orbit")
MAIN_W, MAIN_H = 960, 540
JPEG_Q = 82

_stock_build = C.build_model


def _dressed_build():
    spec = C.build_spec()
    n = blender_props.add_props(spec)
    model = C.scene_look.polish(spec.compile())
    print(f"blender set dressing: {n} visual-only props", flush=True)
    return model


def _cpu_compat():
    """evaluate.rollout moves its input to the literal device "cuda:0".

    That file is hash-pinned (docs/provenance.md), so on machines without a
    GPU we remap the device at the tensor level instead of editing it.
    """
    orig = torch.Tensor.to

    def to(self, *a, **k):
        a = tuple("cpu" if isinstance(x, str) and x.startswith("cuda") else x for x in a)
        return orig(self, *a, **k)

    torch.Tensor.to = to


class Cell:
    """One compiled scene with its env and renderers."""

    def __init__(self, scene):
        C.build_model = _dressed_build if scene == "blender" else _stock_build
        self.env = RandEnv(ConveyorPickEnv())
        C.build_model = _stock_build
        self.r = self.env.real
        m = self.r.model
        self.visap = VisApply(m)
        self.mount = V.Mount(m)
        self.cam_id = {n: int(m.camera(n).id) for n in ("hall", "preview", "overhead")}
        self.rend_ev = self._renderer(128, 128)       # the policy's own camera
        self.rend_main = self._renderer(MAIN_H, MAIN_W)
        self.rend_wrist = self._renderer(256, 256)
        self.free = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, self.free)

    def _renderer(self, h, w):
        rd = mujoco.Renderer(self.r.model, height=h, width=w)
        rd._scene_option.geomgroup[:] = [1] * 6
        rd.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        return rd


class Engine:
    def __init__(self, ckpt=None, ds=None):
        ckpt = ckpt or os.path.join(ROOT, current.CKPT)
        ds = ds or os.path.join(ROOT, current.DS)
        self.dev = gpu_guard.pick_cuda()
        if str(self.dev) == "cpu":
            _cpu_compat()
        ck = torch.load(ckpt, weights_only=True, map_location=self.dev)
        self.stack = int(ck.get("stack", EV.STACK))
        self.qdim = int(ck.get("qdim", 6))
        self.model = Policy(img_ch=self.stack, qdim=self.qdim).to(self.dev)
        load_state(self.model, ck["model"], ckpt)
        self.model.eval()
        with open(os.path.join(ds, "manifest.jsonl")) as fh:
            first = next(json.loads(r) for r in fh if r.strip())
        CS.apply_line_config(first.get("line"))
        self.cells = {}
        self.lock = threading.Lock()
        self.frame = {"main": None, "wrist": None}
        self.frame_seq = 0
        self.wake = threading.Event()
        self.cfg = dict(scene="blender", condition="nominal", camera="hall",
                        smoothing="smooth", servo=True, speed=1.0, seed=4242,
                        orbit=dict(az=140.0, el=-24.0, dist=2.0))
        self.run = dict(running=False, paused=False, restart=False, shutdown=False)
        self.stats = dict(episode=0, picks=0, places=0, phase="idle", t=0.0,
                          gap_mm=0.0, jitter=0.0, fps=0.0, log=[], active=None)
        self._arm_hist = deque(maxlen=24)
        self.info = dict(device=str(self.dev),
                         ckpt=os.path.basename(ckpt),
                         params=int(sum(p.numel() for p in self.model.parameters())))

    # -- API-side helpers (called from HTTP threads) ---------------------
    def get_state(self):
        with self.lock:
            return dict(cfg=json.loads(json.dumps(self.cfg)), run=dict(self.run),
                        stats=json.loads(json.dumps(self.stats)), info=self.info)

    def set_cfg(self, patch):
        with self.lock:
            for k, v in patch.items():
                if k == "scene" and v in SCENES: self.cfg[k] = v
                elif k == "condition" and v in CONDITIONS: self.cfg[k] = v
                elif k == "camera" and v in CAMERAS: self.cfg[k] = v
                elif k == "smoothing" and v in smoothing.PROFILES: self.cfg[k] = v
                elif k == "servo": self.cfg[k] = bool(v)
                elif k == "speed": self.cfg[k] = float(min(8.0, max(0.25, float(v))))
                elif k == "seed": self.cfg[k] = int(v) % (1 << 31)
                elif k == "orbit" and isinstance(v, dict):
                    o = self.cfg["orbit"]
                    o["az"] = float(v.get("az", o["az"])) % 360.0
                    o["el"] = float(min(-5.0, max(-89.0, v.get("el", o["el"]))))
                    o["dist"] = float(min(5.0, max(0.6, v.get("dist", o["dist"]))))

    def control(self, action):
        with self.lock:
            if action == "start":
                self.run.update(running=True, paused=False)
            elif action == "pause":
                self.run["paused"] = True
            elif action == "resume":
                self.run["paused"] = False
            elif action == "stop":
                self.run.update(running=False, restart=True)
            elif action == "restart":
                self.run.update(running=True, paused=False, restart=True)
            elif action == "reset-stats":
                self.stats.update(episode=0, picks=0, places=0, log=[])
        self.wake.set()

    def jpeg(self, which):
        with self.lock:
            return self.frame.get(which), self.frame_seq

    # -- engine thread ----------------------------------------------------
    def _cell(self, scene):
        if scene not in self.cells:
            print(f"building scene: {scene}", flush=True)
            self.cells[scene] = Cell(scene)
        return self.cells[scene]

    def _encode(self, img, q=JPEG_Q):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=q)
        return buf.getvalue()

    def _publish(self, cell, cam):
        r = cell.r
        if cam == "orbit":
            o = self.cfg["orbit"]
            c = cell.free
            c.lookat[:] = (0.0, -0.35, 0.62)
            c.azimuth, c.elevation, c.distance = o["az"], o["el"], o["dist"]
            cell.rend_main.update_scene(r.data, camera=c)
        else:
            cell.rend_main.update_scene(r.data, camera=cell.cam_id[cam])
        main = self._encode(cell.rend_main.render())
        cell.mount.update(r.data)
        cell.rend_wrist.update_scene(r.data, camera=int(r.model.camera("wrist_cam").id))
        V.strip_scene_markers(cell.rend_wrist)
        wrist = self._encode(cell.rend_wrist.render(), 75)
        with self.lock:
            self.frame.update(main=main, wrist=wrist)
            self.frame_seq += 1

    def _episode(self, cell, idx, cond, rng):
        cfg = self.cfg
        vis = sample_vis(rng) if cond == "tierB" else None
        box = EV.BOX_X_CORNERS[idx % len(EV.BOX_X_CORNERS)] if cond == "boxX" else None
        t_wall = dict(last=time.perf_counter(), n=0, t0=time.perf_counter())
        self._arm_hist.clear()

        class Abort(Exception):
            pass

        def on_step(s):
            with self.lock:
                if self.run["restart"] or self.run["shutdown"] or not self.run["running"]:
                    raise Abort()
                paused = self.run["paused"]
                speed, cam = self.cfg["speed"], self.cfg["camera"]
            while paused:
                time.sleep(0.05)
                with self.lock:
                    paused = self.run["paused"]
                    if self.run["restart"] or not self.run["running"]:
                        raise Abort()
                self._publish(cell, cam)
            ph = s["phase"]
            if ph == "policy":
                self._arm_hist.append(np.asarray(s["arm"], float))
                if len(self._arm_hist) >= 3:
                    a = np.array(self._arm_hist)
                    jit = float(np.sqrt((np.diff(a, 2, axis=0) ** 2).sum(1).mean()))
                else:
                    jit = 0.0
                with self.lock:
                    self.stats.update(phase="policy", t=float(s["t"]),
                                      gap_mm=float(s["gap_mm"]), jitter=jit,
                                      picked=bool(s["picked"]), placed=bool(s["placed"]))
            elif ph in ("wait", "handover"):
                with self.lock:
                    self.stats.update(phase=ph, t=0.0)
            # real-time pacing; the belt approach is fast-forwarded
            step = 0.04 / (speed * (4.0 if ph == "wait" else 1.0))
            now = time.perf_counter()
            due = t_wall["last"] + step
            if now < due:
                time.sleep(due - now)
                t_wall["last"] = due
                self._publish(cell, cam)
            else:                       # behind schedule: skip this render
                t_wall["last"] = now
                if ph != "wait" and t_wall["n"] % 3 == 0:
                    self._publish(cell, cam)
            t_wall["n"] += 1
            dt = time.perf_counter() - t_wall["t0"]
            if dt > 1.0:
                with self.lock:
                    self.stats["fps"] = round(t_wall["n"] / dt, 1)
                t_wall.update(t0=time.perf_counter(), n=0)

        stats = {}
        try:
            picked, placed = EV.rollout(
                cell.env, self.model, cell.mount, cell.rend_ev, cell.visap, vis, rng,
                self.stack, None, lrng=np.random.default_rng(777 + idx), box=box,
                on_step=on_step, stats=stats)
        except Abort:
            return None
        return bool(picked), bool(placed), stats

    def run_forever(self):
        rng = None
        idx = 0
        last_cfg = None
        while True:
            with self.lock:
                if self.run["shutdown"]:
                    return
                go = self.run["running"]
            if not go:
                self.wake.wait(0.2); self.wake.clear()
                continue
            with self.lock:
                self.run["restart"] = False
                cfg = json.loads(json.dumps(self.cfg))
            key = (cfg["scene"], cfg["condition"], cfg["seed"])
            if key != last_cfg or rng is None:
                rng, idx, last_cfg = np.random.default_rng(cfg["seed"]), 0, key
                with self.lock:
                    self.stats.update(episode=0, picks=0, places=0, log=[])
            smoothing.install(cfg["smoothing"])
            EV.VISUAL_PICK = EV.VISUAL_PLACE = bool(cfg["servo"])
            cell = self._cell(cfg["scene"])
            with self.lock:
                self.stats["active"] = dict(scene=cfg["scene"], condition=cfg["condition"],
                                            smoothing=cfg["smoothing"], servo=cfg["servo"])
            res = self._episode(cell, idx, cfg["condition"], rng)
            if res is None:
                continue
            picked, placed, st = res
            idx += 1
            with self.lock:
                s = self.stats
                s["episode"] += 1
                s["picks"] += int(picked)
                s["places"] += int(placed)
                s["log"] = (s["log"] + [dict(ep=s["episode"], pick=picked, place=placed,
                                             t=round(float(st.get("end_t", 0.0)), 1))])[-12:]
                s["phase"] = "done"
            time.sleep(1.0)
