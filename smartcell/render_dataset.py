"""Phase 3: deferred wrist-camera rendering for ds_0001 (tier-B visuals).

The collector (Phase 2) stored physics-only streams: one npz per episode with
state and ctrl at the 5 ms physics rate.  This script replays those logged
`act_raw` into the real env and renders `wrist_cam` at 10 Hz over the
manipulation window only - from the first arm motion to 1.0 s after the part
rests in the tote.  Belt-wait rows (up to 30 s at slow belt) are never
rendered.

Observation latency, per the plan, is t - U(0, one obs period): the frame the
policy sees at obs time t was captured up to one full obs period earlier.  It
is implemented exactly that way - frame j renders the logged state row at
episode time (t0 + j/FPS) - u_j, u_j ~ U(0, 1/FPS) - so latency is monotone by
construction and costs nothing at collection time.  At 25 Hz a period is
40 ms, so the worst-case staleness is 2.5x smaller than the 10 Hz dataset's.

Tier-B visual randomization, sampled per episode and restored afterwards:
  lights      8-light rig: brightness x U[0.75, 1.25], warm/cool tint +-10%
  part        m_product rgba multiplied by U[0.6, 1.0]^3 (its colour is a
              texture, so rgba acts as a tint), specular U[0.3, 0.7]
  mount       back +-10 mm, tilt/azim +-4 deg (Phase 6 tests +-40 mm +-8 deg,
              outside this envelope)
  latency     per-frame u_j as above, stored with the images

Output per rendered episode: ds_0001/imgs/ep_XXXXXX_img.npz with
  img (N, 128, 128, 3) uint8, img_t (N,) float32 obs times,
  img_row (N,) int32 source rows, latency (N,) float32,
  vis (json) visual vector.  The manifest row gains "img"/"vis"/"n_img".

SOLL is NOT recorded here, because the collector already stored it: `act_raw`
in the episode npz IS the raw actuator command applied at every physics row
(`buf["act_raw"].append(np.append(ctrl, grip))`), which is exactly the SOLL
value, already in the same radians as `qpos`.  build_cache.py samples it onto
the observation rows, so the IST/SOLL pair costs no re-collection.

Renders IN PLACE into --ds, rewriting that dataset's manifest: to re-render
without disturbing the old frames, copy the npz + manifest into a new dir
first (that is how ds_0005 was made from ds_0004's physics).

Run:  python render_dataset.py --n 24        the verification sample
      python render_dataset.py --all         the full ds_0001
"""
import argparse
import json
import os
import time

import ramcap          # HARD 5 GB cap, see ramcap.py
import numpy as np
import mujoco
from PIL import Image

print("mujoco", mujoco.__version__)
try:
    import torch
    print("torch", torch.__version__)
except Exception as e:                                    # pragma: no cover
    print("torch not imported:", type(e).__name__)

import build_cell as C
import cam_variant as V
import expert as E
import collect_state as CS
from collect_state import RandEnv
from conveyor_env import ConveyorPickEnv
import current

RES = 128
FPS = 25              # Phase 7: 40 ms = exactly 8 physics steps (24 Hz is not)
DT = 0.005
LAT_MAX = 1.0 / FPS   # "the frame used at obs t was captured within one period"
HOME = C.HOME[:5] if hasattr(C, "HOME") else None
if HOME is None:                                          # pragma: no cover
    from conveyor_env import HOME


# The narrow ranges are the MEASURED eval distribution (tier-B in
# evaluate.py calls sample_vis(rng) with wide=False and must not move).
# Training renders default to wide=True: the same channels at 1.5x the
# half-width around the same centres, so the eval distribution is strictly
# INSIDE the training distribution.  Motivation (PROVENANCE "deeper close"):
# ds_0003's close trigger drifted 25 mm up under eval visuals and 37/50
# episodes shut on air - the trigger was load-bearing on a narrow appearance
# prior.  Wider TRAINING visuals (eval unchanged) is the direct fix.
# Draw count per call is identical in both modes, so the latency RNG stream
# that follows in render_episode is unchanged between the two.
VIS_RANGES = dict(
    bright=(0.75, 1.25),
    tint=(0.9, 1.1),
    part_tint=(0.6, 1.0),
    part_spec=(0.3, 0.7),
    dback=(-0.010, 0.010),
    ddeg=(-4.0, 4.0),
    dazim=(-4.0, 4.0),
)
WIDE_K = 1.5


def sample_vis(rng, wide=False):
    def u(lo, hi, n=1):
        if wide:
            c, h = 0.5 * (lo + hi), 0.5 * (hi - lo) * WIDE_K
            lo, hi = c - h, c + h
        v = rng.uniform(lo, hi, n)
        if int(n) == 1:
            return float(np.asarray(v).reshape(-1)[0])
        return [float(x) for x in np.asarray(v).reshape(-1)]
    r = VIS_RANGES
    return dict(
        bright=float(u(*r["bright"])),
        tint=list(u(*r["tint"], 3)),
        part_tint=list(u(*r["part_tint"], 3)),
        part_spec=float(u(*r["part_spec"])),
        dback=float(u(*r["dback"])),
        ddeg=float(u(*r["ddeg"])),
        dazim=float(u(*r["dazim"])),
    )


class VisApply:
    """Writes a visual vector into model arrays and restores them after."""

    def __init__(self, m):
        self.m = m
        self.stock_l = m.light_diffuse.copy()
        self.stock_p = m.mat_rgba[22].copy()
        self.stock_s = float(m.mat_specular[22])

    def apply(self, vis):
        m = self.m
        for i in range(m.nlight):
            base = self.stock_l[i]
            m.light_diffuse[i] = base * vis["bright"] * np.asarray(vis["tint"])
        m.mat_rgba[22, :3] = np.asarray(vis["part_tint"])
        m.mat_specular[22] = vis["part_spec"]

    def restore(self):
        self.m.light_diffuse[:] = self.stock_l
        self.m.mat_rgba[22] = self.stock_p
        self.m.mat_specular[22] = self.stock_s


def window_rows(t, qpos, cubes, active, stop, dt=DT):
    """Manipulation window, matched to the eval handover state.

    Start = the row where the part ARRIVES at the index stop, i.e. the
    collector's wait-loop break; End = the LAST ROW of the episode, which now
    includes the release and the return to home.  Belt-wait rows are never
    rendered: at 25 Hz the window is ~9-10 s = ~240 frames per episode instead
    of the ~570 a whole-episode window needs.

    The 0.4 s SETTLE (the arm held at hover while the drive holds the part) is
    deliberately INSIDE the window.  The eval handover ends at exactly that
    boundary and its frame stack is necessarily filled with settle frames -
    which are bit-identical to each other, because nothing moves.  A window
    that starts after the settle (Phase 7 first attempt) never contained such
    a stack: the model's only "static for 280 ms" training pattern was the
    pre-close park, so it read the settle stack as "parked -> close now" and
    drove to the tote with the grip shut at the FIRST decision (measured:
    probe_eval0_25.py).  With the settle inside the window the same static
    stack is a training input whose ground truth is "hold open, hover", and
    the only difference from the park is what the IMAGE shows (the part 0.16 m
    away vs between the jaws) - which is the cue the policy is supposed to use.
    """
    arr = ((cubes[:, 0, 0] > stop - 0.002)
           & (np.abs(cubes[:, 0, 1] - C.BELT_Y) < 0.06))
    k0 = int(np.argmax(arr)) if arr.any() else 0
    # End = END OF EPISODE, not "1 s after the drop".  The collector now returns
    # the arm home after the release, and that return is the ONLY demonstration
    # of what the policy should do once the task is complete.  Truncating the
    # window there left the post-release behaviour untrained while eval runs a
    # 30 s horizon, which is exactly the "does not know it should go back"
    # failure.  Measured on ds_0005: the old rule covered 43.9% of the episode.
    k1 = len(t) - 1
    return k0, k1


FULL_EPISODE = False          # set by --full: render the whole episode


def render_episode(env, mount, rend, visap, row, vis, rng, ds):
    r = env.real
    z = np.load(os.path.join(ds, row["npz"]))
    act_raw = z["act_raw"].astype(np.float64)
    t, qpos, cubes = z["t"].astype(np.float64), z["qpos"], z["cubes"]
    vec = row["vector"] if isinstance(row["vector"], dict) else json.loads(row["vector"])
    env.vec = {**vec, "jaw_yaw_vec": np.array(vec["jaw_yaw_vec"]),
               "aim_xy": np.array(vec["aim_xy"])}
    # the replay below re-runs the belt logic, so the per-episode indexing stop
    # has to be installed or the part would not rest where it did at collection.
    # Both patches go in BEFORE window_rows: its end-of-window check reads
    # build_cell.TOTE_INNER at call time, so the tote patch moves it too.
    CS.apply_line_config(row.get("line"))
    stop = CS.apply_stop(env.vec)
    CS.apply_tote(env.vec, env)
    k0, k1 = window_rows(t, qpos, cubes, z["active"], stop)
    if FULL_EPISODE:
        # FROM ROW 0, so the episode begins at the WAIT POSE.
        #
        # window_rows starts at the row where the part reaches the index stop,
        # which drops the whole belt wait -- the part of the episode the user
        # actually specified, and the part where the arm is standing at "the
        # main waiting spot".  The inspector showed a clip that began mid-pick
        # and the policy never saw a single wait frame.  The wait is now
        # rendered and cached like everything else.
        #
        # k1 is left alone: it is already the last row of the episode, which
        # includes the release and the return to the start pose.
        k0 = 0
    CS.CTX.pop("hook", None)
    r._arm_lo = r.model.actuator_ctrlrange[r._arm_aid, 0]
    r._arm_hi = r.model.actuator_ctrlrange[r._arm_aid, 1]
    visap.apply(vis)
    # mount jitter = attribute assignment on the ONE Mount instance: update()
    # recomputes cam_pos/cam_quat from back/deg/azim every frame, and a second
    # Mount(model) would trip check_mount (the first instance world-parents
    # the camera for the life of the process - the documented 3.14.0 fix)
    mount.back = V.DEFAULT["back"] + vis["dback"]
    mount.deg = V.DEFAULT["deg"] + vis["ddeg"]
    # DEFAULT + jitter, NOT the jitter alone.  `dazim` is a deviation; writing it
    # straight into azim was invisible while DEFAULT["azim"] was 0.0 (0 +- 4 and
    # +- 4 are the same set), but DEFAULT is now 180 (cam_variant, after the
    # near-plane fix re-derived the mount) and assigning the raw deviation would
    # have rendered every training frame at azim +-4 instead of 180 +-4 - i.e.
    # the camera pointing the wrong way round for the whole dataset.
    mount.azim = V.DEFAULT["azim"] + vis["dazim"]
    env.reset(seed=row["seed"])
    # the episode's start pose.  reset() writes HOME, and these episodes begin
    # at the user's wait pose, so without this the render replays a different
    # episode than the one on disk: measured `replay_dev 1.9e+00` on the
    # collector's own check when the start pose moved and this was missing.
    CS.install_wait_pose(env)
    t0 = t[k0]
    n = int((t[k1] - t0) * FPS) + 1
    lat = rng.uniform(0.0, LAT_MAX - 1e-4, n)
    rows_want = np.round((t0 + np.arange(n) / FPS - lat) / DT).astype(int)
    rows_want = np.clip(rows_want, 0, len(act_raw) - 1)
    # u_j = 0.1 next to u_j = 0 can round two obs to the same physics row;
    # nudge such ties forward by one row (5 ms) rather than emit duplicates
    for j in range(1, n):
        if rows_want[j] <= rows_want[j - 1]:
            rows_want[j] = rows_want[j - 1] + 1
    rows_want = np.clip(rows_want, 0, len(act_raw) - 1)
    assert np.all(np.diff(rows_want) > 0), "latency broke monotonicity"

    imgs = np.empty((n, RES, RES, 3), dtype=np.uint8)
    t_replay = t_render = 0.0
    cam_id = int(r.model.camera("wrist_cam").id)
    done = 0
    for j in range(len(act_raw)):
        ts = time.perf_counter()
        E.sim(env, r.model.opt.timestep, arm=act_raw[j, :5],
              grip=float(act_raw[j, 5]))
        t_replay += time.perf_counter() - ts
        while done < n and j >= rows_want[done]:
            tr = time.perf_counter()
            mount.update(r.data)
            rend.update_scene(r.data, camera=cam_id)
            # mjv_updateScene appends a translucent magenta DEBUG sphere for
            # light 2, 67 mm in front of the wrist lens; it is in every frame
            # otherwise.  See cam_variant.strip_scene_markers.
            V.strip_scene_markers(rend)
            imgs[done] = rend.render()
            t_render += time.perf_counter() - tr
            done += 1
        if done >= n:
            break
    assert done == n, (done, n)
    visap.restore()
    return dict(img=imgs, img_t=(t0 + np.arange(n) / FPS).astype(np.float32),
                img_row=rows_want.astype(np.int32),
                latency=lat.astype(np.float32), vis=vis,
                t_replay=t_replay, t_render=t_render,
                window_s=float(t[k1] - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--ep", type=int, nargs="*", default=None)
    ap.add_argument("--ds", default=current.DS)
    ap.add_argument("--all", action="store_true", default=True,
                    help="default True: render EVERY episode. "
                         "Pass --sample N to render a spread of N instead.")
    ap.add_argument("--sample", type=int, default=None,
                    help="render only a spread of N episodes")
    ap.add_argument("--no-full", dest="full", action="store_false",
                    help="render from row 0, so the frames start at the wait "
                         "pose instead of at the moment the part reaches the "
                         "stop (costs ~2.3x the frames: the belt wait is "
                         "~9 s of an ~18 s episode at 0.08 m/s)")
    ap.add_argument("--vis", default="wide", choices=("wide", "narrow"),
                    help="training renders default to the WIDE visual "
                         "distribution; 'narrow' reproduces the eval ones")
    args = ap.parse_args()
    global FULL_EPISODE
    FULL_EPISODE = bool(args.full)
    print(f"visual distribution: {args.vis}"
          + ("  |  FULL EPISODE from the wait pose" if FULL_EPISODE else
             "  |  manipulation window only"))

    rows = [json.loads(l) for l in open(os.path.join(args.ds, "manifest.jsonl"))]
    ok = [r for r in rows if r["ok"]]
    if args.ep:
        pick = [r for r in ok if r["ep_id"] in args.ep]
    elif args.sample:
        # the OLD way: a spread of N.  --all is the default now, so this only
        # runs when --sample is asked for explicitly.
        pick = [ok[i * len(ok) // args.sample] for i in range(args.sample)]
    else:
        pick = ok
    print(f"rendering {len(pick)} of {len(ok)} episodes from {args.ds}"
          + ("  (FULL EPISODE)" if FULL_EPISODE else "  (window only)"))

    env = RandEnv(ConveyorPickEnv())
    m = env.real.model
    visap = VisApply(m)
    mount = V.Mount(m)
    rend = mujoco.Renderer(m, height=RES, width=RES)
    rend._scene_option.geomgroup[:] = [1] * 6
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    os.makedirs(os.path.join(args.ds, "imgs"), exist_ok=True)

    tr = tren = 0.0
    frames = []
    n_img = 0
    for i, row in enumerate(pick):
        rng = np.random.default_rng(row["seed"] ^ 0x5EED)
        vis = sample_vis(rng, wide=(args.vis == "wide"))
        out = render_episode(env, mount, rend, visap, row, vis, rng, args.ds)
        path = os.path.join(args.ds, "imgs", f"ep_{row['ep_id']:06d}_img.npz")
        np.savez_compressed(
            path, img=out["img"], img_t=out["img_t"], img_row=out["img_row"],
            latency=out["latency"], vis=np.array(json.dumps(out["vis"])))
        row["img"] = f"imgs/ep_{row['ep_id']:06d}_img.npz"
        row["vis"] = out["vis"]
        row["n_img"] = int(len(out["img"]))
        row["window_s"] = round(out["window_s"], 2)
        row["vis_mode"] = args.vis
        # the pixels now come from the CURRENT build (e.g. boxmark inlay):
        # stamp the code hash so a re-rendered manifest never claims the
        # old build that collected it
        row["code"] = CS.code_hash()
        tr += out["t_replay"]
        tren += out["t_render"]
        frames.append(len(out["img"]))
        n_img += len(out["img"])
        print(f"  [{i + 1:3d}/{len(pick)}] ep {row['ep_id']:3d}  "
              f"{len(out['img']):3d} frames  window {out['window_s']:5.2f}s  "
              f"render {out['t_render'] / max(len(out['img']), 1) * 1000:5.2f} ms/fr  "
              f"lat max {out['latency'].max() * 1000:5.1f} ms", flush=True)

    # rewrite the manifest with the image fields
    done = {r["ep_id"] for r in pick}
    tmp = os.path.join(args.ds, "manifest.jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, os.path.join(args.ds, "manifest.jsonl"))

    fr = np.asarray(frames, float)
    per_ep = (tr + tren) / max(len(pick), 1)
    print(f"\nframes {n_img}  replay {tr:.1f}s  render {tren:.1f}s")
    print(f"mean render {tren / max(n_img, 1) * 1000:.2f} ms/frame  "
          f"mean replay {tr / max(n_img, 1) * 1000:.2f} ms/frame  "
          f"(replay steps 5x denser than frames)")
    z0 = np.load(os.path.join(args.ds, "imgs",
                              f"ep_{pick[0]['ep_id']:06d}_img.npz"))
    mb = os.path.getsize(os.path.join(args.ds, "imgs",
                                      f"ep_{pick[0]['ep_id']:06d}_img.npz")) / 1e6
    print(f"sample img npz: {z0['img'].shape} {z0['img'].dtype}, "
          f"{mb:.2f} MB on disk")
    # per-episode cost is (replay+render) per rendered episode, measured above;
    # frames per episode vary with the manipulation window, hence per_ep
    print(f"per-episode cost: {per_ep:.2f} s (replay+render, this sample)")
    print(f"projection, all {len(ok)} eps: {per_ep * len(ok) / 60:.1f} min "
          f"single-proc, ~{per_ep * len(ok) / 60 / 4:.1f} min on 4 workers")
    print(f"projection, 3000 eps x 2 passes: "
          f"{per_ep * 2 * 3000 / 60:.0f} min single-proc, "
          f"~{per_ep * 2 * 3000 / 60 / 4:.0f} min on 4 workers")
    print(f"disk: ~{mb * len(ok) / 1000:.2f} GB for {len(ok)} eps, "
          f"~{mb * 3000 * 2 / 1000:.1f} GB for 3000 eps x 2 passes")
    env.real.close()
    rend.close()


if __name__ == "__main__":
    main()
