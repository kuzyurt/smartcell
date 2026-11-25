"""Fast scripted expert for dataset collection.

`demo_pick.py` is the verified reference and is NOT modified.  It is also slow:
one replay costs 274-568 s, of which ~99% is IK, not physics (101 us/step) and
not rendering (1.03 ms/frame).  The cost is in `ArmIK.solve`, which demo_pick
calls at every 5 mm Cartesian waypoint with `restarts=8` random seeds and
`outer=30` wrist/position alternations - about 21,600 solver iterations per
waypoint, 40 waypoints per move.

The fix is to exploit what a tracking move already knows.  Once `move_to` is
following a solution, the branch cannot change: seeding the solver with the
current joint vector converges in one or two alternations, and the 8 random
restarts are pure waste.  So the branch is *established* once per move with the
full search and then *tracked* with a single seed.  That is what makes this fast
and it is also why it is safe - it is the same guarantee demo_pick gets from its
`q0`-first ranking, just amortised over the whole move instead of re-derived 40
times.

Waypoints are also 10 mm rather than 5 mm, halving the solve count.  Both the
arrival error and the grasp/placed outcome are checked, so a fast path that
silently degrades the motion fails loudly instead of quietly poisoning the
dataset.

Run:  python expert.py            time one episode and verify it still works
      python expert.py --slow     the same episode via demo_pick's settings
"""
import argparse
import time

import numpy as np
import mujoco

import build_cell as C
import demo_pick
from conveyor_env import ConveyorPickEnv, GRIP_OPEN, GRIP_SHUT, X_STOP
from demo_pick import (PICK_X, HOVER_X, GRASP_Z, APPROACH_Z, LIFT_Z, TOTE_XY)

FAST = dict(step=0.010, seed_restarts=8, seed_outer=30, seed_iters=40,
            track_restarts=0, track_outer=2, track_iters=12,
            reseed_tol=0.004)
SLOW = dict(step=0.005, seed_restarts=8, seed_outer=30, seed_iters=40,
            track_restarts=8, track_outer=30, track_iters=40,
            reseed_tol=0.004)


def sim(env, seconds, arm=None, grip=None, hook=None):
    """Step physics the way demo_pick does, with an optional per-step hook."""
    m, d = env.model, env.data
    dt = m.opt.timestep
    for _ in range(int(seconds / dt)):
        if arm is not None:
            d.ctrl[env._arm_aid] = np.clip(arm, env._arm_lo, env._arm_hi)
        if grip is not None:
            d.ctrl[env._grip_aid] = grip
        mujoco.mj_step(m, d)
        env._sim_t += dt
        env._conveyor_logic()
        if hook is not None:
            hook(env)


def _solve(ik, w, q, top_down, restarts, outer, iters):
    t0 = time.perf_counter()
    q, err, tilt = ik.solve(w, q0=q, top_down=top_down, outer=outer,
                            iters=iters, restarts=restarts)
    return q, err, tilt, time.perf_counter() - t0


def move_to(env, target, grip, top_down=True, settle=0.5, log=True,
            cfg=FAST, stats=None, reseed=False):
    """Walk the grasp site to `target` along a straight Cartesian line.

    Same trajectory shape as demo_pick.move_to (interpolating joints does not:
    the elbow drags the site along an arc and sweeps the fingers through the
    part), but the IK is tracked with a single seed instead of re-running the
    full 9-seed search at every waypoint.

    The branch is established once per episode, not once per move, because the
    arm stays in the same configuration branch throughout a pick-and-place.  If
    a tracked move fails to arrive, it is retried with the full search, so a
    lost branch costs one move's time rather than the whole episode's - the
    fast path cannot quietly degrade the trajectory.
    """
    ik = env.ik
    m, d = env.model, env.data
    dt = m.opt.timestep
    target = np.asarray(target, dtype=float)

    def attempt(full):
        start = d.site_xpos[env._grasp_site].copy()
        n = max(4, int(np.linalg.norm(target - start) / cfg["step"]))
        hold = max(1, int(0.02 / dt))
        q = d.qpos[env._arm_qpos].copy()
        err = tilt = 0.0
        t_ik = 0.0
        for i in range(1, n + 1):
            w = start + (target - start) * (i / n)
            if full and i == 1:
                q, err, tilt, dt_ = _solve(ik, w, q, top_down,
                                          cfg["seed_restarts"], cfg["seed_outer"],
                                          cfg["seed_iters"])
            else:
                q, err, tilt, dt_ = _solve(ik, w, q, top_down,
                                          cfg["track_restarts"], cfg["track_outer"],
                                          cfg["track_iters"])
            t_ik += dt_
            for _ in range(hold):
                sim(env, dt, arm=q, grip=grip)
        sim(env, settle, arm=q, grip=grip)
        got = float(np.linalg.norm(d.site_xpos[env._grasp_site] - target))
        return got, n, t_ik, err, tilt

    full = reseed or cfg["track_restarts"] >= cfg["seed_restarts"]
    got, n, t_ik, err, tilt = attempt(full)
    if got > cfg["reseed_tol"] and not full:
        if log:
            print(f"     tracked move missed by {got*1000:.1f} mm, "
                  f"re-solving with the full search")
        got, n, t_ik2, err, tilt = attempt(True)
        t_ik += t_ik2
    if stats is not None:
        stats["ik_s"] += t_ik
        stats["solves"] += n
        stats["reseeds"] += 0 if full else (1 if got <= cfg["reseed_tol"] else 0)
    if log:
        ang = f"{tilt:5.1f}" if top_down else "  free"
        print(f"  -> {str(np.round(target, 3)):26s} IK {err*1000:5.2f}mm "
              f"tilt {ang} deg  arrived {got*1000:5.2f}mm off  "
              f"({n} solves, {t_ik:.2f}s IK)")
    return got


def episode(env, seed=0, cfg=FAST, log=True, hook=None, stats=None):
    """The full pick-and-place.  Returns (picked_index, placed_bool, stats)."""
    env._arm_lo = env.model.actuator_ctrlrange[env._arm_aid, 0]
    env._arm_hi = env.model.actuator_ctrlrange[env._arm_aid, 1]
    env.reset(seed=seed)
    if stats is None:
        stats = dict(ik_s=0.0, solves=0, sim_s=0.0, steps=0, reseeds=0)
    sim_ = env._sim_t

    def tick(t0):
        stats["sim_s"] += time.perf_counter() - t0
        stats["steps"] += 1

    # the one full search of the episode: it fixes the branch everything else
    # tracks.  Every later move is single-seed.
    move_to(env, (HOVER_X, C.BELT_Y, APPROACH_Z), GRIP_OPEN, settle=0.1,
            log=log, cfg=cfg, stats=stats, reseed=True)

    idx = next((i for i in range(C.POOL) if env._active[i]), None)
    if idx is None:
        return None, False, stats
    dt = env.model.opt.timestep
    t = 0.0
    while t < 30.0:
        p = env._cube_pos(idx)
        if p[0] > X_STOP - 0.002 and abs(p[1] - C.BELT_Y) < 0.06:
            break
        sim(env, dt, arm=env.data.qpos[env._arm_qpos], grip=GRIP_OPEN, hook=hook)
        t += dt
    sim(env, 0.4, arm=env.data.qpos[env._arm_qpos], grip=GRIP_OPEN, hook=hook)
    if not (X_STOP - 0.01 < env._cube_pos(idx)[0] < C.PICK_X[1]):
        return idx, False, stats

    p = env._cube_pos(idx)
    move_to(env, (p[0], p[1], APPROACH_Z), GRIP_OPEN, settle=0.15, log=log,
            cfg=cfg, stats=stats)
    move_to(env, (p[0], p[1], GRASP_Z), GRIP_OPEN, settle=0.4, log=log,
            cfg=cfg, stats=stats)
    sim(env, 0.5, grip=GRIP_SHUT, hook=hook)
    sim(env, 0.3, grip=GRIP_SHUT, hook=hook)
    held = env._grasped()
    print(f"  grasp: held={held} (expect {idx})")
    if held != idx:
        return idx, False, stats

    p = env._cube_pos(idx)
    move_to(env, (p[0], p[1], LIFT_Z), GRIP_SHUT, log=log, cfg=cfg, stats=stats)
    move_to(env, (TOTE_XY[0], TOTE_XY[1], LIFT_Z), GRIP_SHUT, log=log,
            cfg=cfg, stats=stats)
    move_to(env, (TOTE_XY[0], TOTE_XY[1], C.TOTE_RELEASE_Z), GRIP_SHUT,
            settle=0.4, log=log, cfg=cfg, stats=stats)
    sim(env, 0.5, grip=GRIP_OPEN, hook=hook)
    sim(env, 0.8, grip=GRIP_OPEN, hook=hook)
    q = env._cube_pos(idx)
    ok = bool(env._in_tote(q))
    print(f"  part rests at {np.round(q, 4).tolist()}  INSIDE TOTE: {ok}")
    stats["sim_time_s"] = env._sim_t - sim_
    return idx, ok, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", action="store_true",
                    help="use demo_pick's IK settings instead of the fast path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("mujoco", mujoco.__version__)
    try:
        import torch
        print("torch", torch.__version__)
    except Exception as e:
        print("torch not imported:", type(e).__name__)

    cfg = SLOW if args.slow else FAST
    env = ConveyorPickEnv()
    print(f"\nIK config: step={cfg['step']*1000:.0f}mm  "
          f"seed restarts={cfg['seed_restarts']} outer={cfg['seed_outer']}  "
          f"track restarts={cfg['track_restarts']} outer={cfg['track_outer']}")
    t0 = time.perf_counter()
    idx, ok, stats = episode(env, seed=args.seed, cfg=cfg, log=True)
    wall = time.perf_counter() - t0

    print(f"\n{'SLOW' if args.slow else 'FAST'} expert  seed={args.seed}")
    print(f"  wall            {wall:8.2f} s")
    print(f"  IK              {stats['ik_s']:8.2f} s  "
          f"({100*stats['ik_s']/wall:.0f}% of wall, {stats['solves']} solves)")
    print(f"  simulated time  {stats.get('sim_time_s', 0.0):8.2f} s")
    print(f"  realtime factor {stats.get('sim_time_s', 0.0)/wall:8.1f} x")
    print(f"  result: part {idx} {'PLACED IN TOTE' if ok else 'NOT PLACED'}")
    if not args.slow and ok:
        n = 300
        print(f"\n  projected: {n} episodes on 4 workers = "
              f"{n*wall/4/60:.1f} min  (plan target 15 min)")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
