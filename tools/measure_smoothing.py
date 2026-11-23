"""Measure arm smoothness and success for a smoothing profile.

    python tools/measure_smoothing.py --profile '{"ramp":1.0}' --n 8 --out m.json

Joint angles are sampled at every physics step (200 Hz) while the frozen
policy runs the Blender-dressed or stock cell on fixed seeds. Reports RMS
jerk / acceleration of the *actual* joints and the pick/place outcome.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "smartcell"))

import numpy as np

from web.engine import Engine, EV, Cell, sample_vis
import smoothing

DT = 0.005


def analyse(q):
    """q: [T,5] joint angles at 200 Hz -> smoothness numbers (rad, s)."""
    v = np.diff(q, axis=0) / DT
    a = np.diff(v, axis=0) / DT
    j = np.diff(a, axis=0) / DT
    k = np.ones(25) / 25.0                      # 125 ms moving average
    sm = np.stack([np.convolve(q[:, i], k, "same") for i in range(q.shape[1])], 1)
    hf = (q - sm)[25:-25]
    sgn = np.sign(v[:, :])
    return dict(
        rms_acc=float(np.sqrt((a ** 2).sum(1).mean())),
        rms_jerk=float(np.sqrt((j ** 2).sum(1).mean())),
        hf_rms=float(np.sqrt((hf ** 2).sum(1).mean())),
        path=float(np.abs(np.diff(q, axis=0)).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="off")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scene", default="stock")
    ap.add_argument("--cond", default="nominal")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prof = json.loads(a.profile) if a.profile.startswith("{") else a.profile
    eng = Engine()
    cell = Cell(a.scene)
    p = smoothing.install(prof)
    EV.VISUAL_PICK = EV.VISUAL_PLACE = True
    rng = np.random.default_rng(a.seed)
    for _ in range(a.start):
        if a.cond == "tierB":
            sample_vis(rng)
        rng.integers(1 << 31)
    rows = []
    for i in range(a.start, a.start + a.n):
        smoothing.RECORD = []
        st = {}
        vis = sample_vis(rng) if a.cond == "tierB" else None
        box = EV.BOX_X_CORNERS[i % len(EV.BOX_X_CORNERS)] if a.cond == "boxX" else None
        picked, placed = EV.rollout(cell.env, eng.model, cell.mount, cell.rend_ev,
                                    cell.visap, vis, rng, eng.stack, None,
                                    lrng=np.random.default_rng(777 + i), box=box, stats=st)
        q = np.array(smoothing.RECORD)
        # analyse the motion phase only: from first policy step on
        row = analyse(q[len(q) // 8:])
        row.update(i=i, picked=bool(picked), placed=bool(placed),
                   pick_t=st.get("pick_t"), place_t=st.get("place_t"), end_t=st.get("end_t"))
        rows.append(row)
        print(f"[{i}] pick {int(picked)} place {int(placed)} jerk {row['rms_jerk']:.0f} "
              f"hf {row['hf_rms']:.4f}", flush=True)
        json.dump(dict(profile=p, scene=a.scene, rows=rows), open(a.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
