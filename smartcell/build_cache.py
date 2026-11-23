"""Cache builder: imgs/*.npz + ds episode npz -> cache/ep_XXXXXX.npz.

The original builder is gone from the tree, so this one is written from the
alignment that probe_cache_align.py could measure and is VALIDATED against the
surviving cache: `--verify` builds the four episodes in the 10 Hz grid
(fps 10, entry 0.05 s) from ds_0001 and prints the difference against
cache_10hz/ instead of writing anything.  Same code then builds the 25 Hz
cache for ds_0002 with fps 25 / entry 0.04 s.

Definition of cache row j (obs time t_j = img_t[j]):
  g[j]       area 128->96, mean of the three channels, uint8 (matches
             evaluate.current_frame's decode exactly)
  q[j]       IST: qpos (5 joints + gripper) at the state row nearest t_j.  The
             IMAGE is deliberately u ~ U(0, 1/fps) STALE relative to that row
             (Phase 3 latency, kept in img_row/latency), the proprio and the
             action targets are not - which is what evaluate.py reproduces by
             shifting the frame stack back 0..1 slots.
  soll[j]    SOLL: the raw actuator command ("act_raw") at the SAME row as
             q[j].  Same units as q[j] (radians), so soll - q is the tracking
             error.  For the gripper that error is a causal "is something
             between the jaws" sensor: measured on the expert, a settled close
             stalls at |SOLL-IST| = 0.18829 holding the 32 mm part versus
             0.00000-0.00152 with the jaws empty.  No re-collection was needed
             - the collector already stored act_raw; it simply was not being
             carried into the cache.  q and soll are concatenated into the
             model's 12-wide proprio by dataset.py.
  chunks[j]  the expert's own normalised ctrl for entries k = 0..7, entry k at
             state row (row + k * entry/dt); entry 0 is the action commanded AT
             the observation, i.e. the first action the policy should apply.
  t, lat, ep  obs times, per-frame latency, episode id.

Run:  python build_cache.py --ds ds_0001 --fps 10 --entry 0.05 --verify
      python build_cache.py --ds ds_0002 --fps 25 --entry 0.04 --out cache
"""
import argparse
import os
import time

import ramcap          # HARD 5 GB cap, see ramcap.py
import numpy as np
import torch
import torch.nn.functional as F

print("torch", torch.__version__)

import collect_state as CS       # normalised_actions + the jaw-centring patch
import current

RES_CAM = 128
RES = 96


def gray96(imgs):
    """(N,128,128,3) uint8 -> (N,96,96) uint8, the eval decode's exact path."""
    t = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(RES, RES), mode="area").mean(1)
    return t.numpy()


def build_ep(env, ds, ep, fps, entry, out_dir=None, verify_dir=None):
    zi = np.load(os.path.join(ds, "imgs", f"ep_{ep:06d}_img.npz"))
    z = np.load(os.path.join(ds, f"ep_{ep:06d}.npz"))
    imgs, img_t = zi["img"], zi["img_t"].astype(np.float64)
    # --fps is not decorative (Act 19): the obs grid comes from the frames'
    # own img_t, so assert the frames really are at that rate instead of
    # silently building a differently-spaced grid.
    if len(img_t) > 1:
        spacing = float(np.median(np.diff(img_t)))
        want = 1.0 / float(fps)
        assert abs(spacing - want) < 0.0025, (
            f"{ds}/imgs/ep_{ep:06d}: frames are {spacing * 1000:.1f} ms "
            f"apart, --fps {fps:g} expects {want * 1000:.1f} ms")
    t, qpos = z["t"].astype(np.float64), z["qpos"]
    act = z["act"]                      # already normalised by the collector
    act_raw = z["act_raw"]              # raw actuator commands = SOLL
    dt = float(env.real.model.opt.timestep)
    n = len(imgs)
    stride = int(round(float(entry) / dt))
    rows = np.array([int(np.argmin(np.abs(t - tj))) for tj in img_t])

    g = np.clip(np.rint(gray96(imgs)), 0, 255).astype(np.uint8)
    q = qpos[rows].astype(np.float32)
    # act_raw has one row FEWER than t (a command is applied and then the sim
    # steps, so the final state row has no command of its own); clip like the
    # chunk builder does rather than read off the end
    soll = act_raw[np.clip(rows, 0, len(act_raw) - 1)].astype(np.float32)
    ch = np.empty((n, 8, 6), dtype=np.float32)
    for j in range(n):
        idx = np.clip(rows[j] + stride * np.arange(8), 0, len(act) - 1)
        ch[j] = act[idx]
    out = dict(g=g, chunks=ch, q=q, soll=soll, t=img_t.astype(np.float32),
               lat=zi["latency"].astype(np.float32),
               img_row=rows.astype(np.int32), ep=np.int64(ep),
               ds=np.array(os.path.basename(os.path.normpath(ds))))

    if verify_dir is not None:
        # The surviving 10 Hz cache covers a SUB-RANGE of the frames in the
        # imgs npz (the old render pass wrote more frames than the cache used),
        # so align by observation time and report both the alignment and the
        # per-field difference at the matched frames.
        oc = np.load(os.path.join(verify_dir, f"ep_{ep:06d}.npz"))
        j = np.array([int(np.argmin(np.abs(img_t - tj))) for tj in oc["t"]])
        k = np.array([int(np.argmin(np.abs(oc["t"] - tj))) for tj in img_t])
        dg = np.abs(oc["g"].astype(np.int32) - g[j].astype(np.int32))
        dq = np.abs(oc["q"] - q[j]).max()
        dc = np.abs(oc["chunks"] - ch[j])
        dt_ = np.abs(oc["t"] - img_t[j]).max()
        print(f"  ep {ep:3d}  imgs {n} (t {img_t[0]:.3f}..{img_t[-1]:.3f})  "
              f"cache {len(oc['g'])} rows -> imgs idx {j[0]}..{j[-1]} "
              f"(step {np.median(np.diff(j)):.0f})  "
              f"g mean|d| {dg.mean():.3f} exact {np.mean(dg == 0) * 100:.1f}%  "
              f"q max|d| {dq:.2e}  chunks max|d| {dc.max():.2e} "
              f"(frac>0.02 {np.mean(dc > 0.02) * 100:.2f}%, frac>0.2 "
              f"{np.mean(dc > 0.2) * 100:.2f}%)  t max|d| {dt_:.2e}  "
              f"(imgs idx of cache t0 {j[0]}, of cache t-1 {j[-1]})")
        if k[j[0]] > 0:
            print(f"      frames not in the cache: {k[j[0]]} before its t0, "
                  f"{n - 1 - k[j[-1]]} after its last row")
        return out
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f"ep_{ep:06d}.npz"), **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=current.DS)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--entry", type=float, default=0.04, help="chunk entry, s")
    ap.add_argument("--out", default=current.CACHE)
    ap.add_argument("--verify", action="store_true",
                    help="compare against --out instead of writing")
    ap.add_argument("--ep", type=int, nargs="*", default=None)
    args = ap.parse_args()

    from conveyor_env import ConveyorPickEnv
    from collect_state import RandEnv
    env = RandEnv(ConveyorPickEnv())
    eps = args.ep if args.ep is not None else sorted(
        int(f[3:9]) for f in os.listdir(os.path.join(args.ds, "imgs"))
        if f.endswith("_img.npz"))
    print(f"{'verifying' if args.verify else 'building'} {len(eps)} episodes from"
          f" {args.ds} at {args.fps:g} Hz, chunk entry {args.entry * 1000:.0f} ms"
          f" ({int(round(args.entry / 0.005))} physics rows)")
    t0 = time.perf_counter()
    tot = 0
    for i, ep in enumerate(eps):
        out = build_ep(env, args.ds, ep, args.fps, args.entry,
                       out_dir=None if args.verify else args.out,
                       verify_dir=args.out if args.verify else None)
        tot += len(out["g"])
        if not args.verify and (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(eps)}] {tot} frames  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    if args.verify:
        print(f"verify done in {time.perf_counter() - t0:.0f}s (nothing written)")
    else:
        print(f"done: {len(eps)} eps, {tot} frames, "
              f"{time.perf_counter() - t0:.0f}s -> {args.out}")
    env.real.close()


if __name__ == "__main__":
    main()
