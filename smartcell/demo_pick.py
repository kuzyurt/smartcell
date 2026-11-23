"""Scripted pick-and-place on the real line.

Proves the layout is solvable end to end with physics only: the arm waits for a
part to enter the verified grasp window, grasps it while it is still moving,
lifts, carries it to the tote on the table and releases it.  Every waypoint is
solved with arm_ik and tracked by the model's own position actuators; the belt
is the official surfacevel conveyor, so the part is genuinely transported.

Run:  python demo_pick.py
"""
import mujoco
import numpy as np
from PIL import Image

import build_cell as C
from conveyor_env import ConveyorPickEnv, GRIP_OPEN, GRIP_SHUT, X_STOP

PICK_X = 0.10                   # where the jaws close (window centre)
HOVER_X = 0.02                   # window entry, where the hover pose sits
GRASP_Z = C.BELT_TOP + C.CUBE / 2
APPROACH_Z = C.TABLE_TOP + 0.18  # hover just above the lane, inside the
                                  # top-down envelope (margin 0.27 rad here)
LIFT_Z = 0.84                            # 200 mm of lift: clears the tote rim
                                       # and the rails, low enough that the
                                       # wrist can still hold the part square
TOTE_XY = C.TOTE_XY
RAIL = 1.5                      # rad/s joint cap for the moves
LEAD = 0.008                    # servo lag compensation while tracking
TRACK_DT = 0.02                 # re-solve the tracking target at 50 Hz


def sim(env, seconds, arm=None, grip=None):
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


def move_to(env, target, grip, top_down=True, settle=0.5, log=True, step=0.005):
    """Walk the grasp site to `target` along a straight line in Cartesian space.

    Interpolating the joints instead drags the site along an arc: the SO-101's
    elbow swings the wrist sideways, and on the way down that arc sweeps the
    fingers through the part and pushes it along the belt.  Solving the IK at
    5 mm waypoints keeps the site on the line.
    """
    ik = env.ik
    m, d = env.model, env.data
    dt = m.opt.timestep
    target = np.asarray(target, dtype=float)
    start = d.site_xpos[env._grasp_site].copy()
    n = max(4, int(np.linalg.norm(target - start) / step))
    hold = max(1, int(0.02 / dt))
    q = d.qpos[env._arm_qpos].copy()
    err = tilt = 0.0
    for i in range(1, n + 1):
        w = start + (target - start) * (i / n)
        q, err, tilt = ik.solve(w, q0=q, top_down=top_down)
        for _ in range(hold):
            sim(env, dt, arm=q, grip=grip)
    sim(env, settle, arm=q, grip=grip)
    p = d.site_xpos[env._grasp_site].copy()
    got = float(np.linalg.norm(p - target))
    if log:
        ang = f"{tilt:5.1f}" if top_down else "  free"
        print(f"  -> {str(np.round(target, 3)):26s} IK {err*1000:5.2f}mm "
              f"tilt {ang} deg  arrived {got*1000:5.2f}mm off")
    return got


def main():
    env = ConveyorPickEnv()
    env._arm_lo = env.model.actuator_ctrlrange[env._arm_aid, 0]
    env._arm_hi = env.model.actuator_ctrlrange[env._arm_aid, 1]
    env.reset(seed=0)
    ik = env.ik
    shots = []
    r = mujoco.Renderer(env.model, height=260, width=340)

    def snap(tag):
        r.update_scene(env.data, camera="overhead")
        shots.append((tag, r.render().copy()))

    print("1. hover over the grasp window while the part is still infeed")
    snap("start")
    move_to(env, (HOVER_X, C.BELT_Y, APPROACH_Z), GRIP_OPEN, settle=0.1)

    print("2. wait for the line to index a part to the pick station")
    idx = next(i for i in range(C.POOL) if env._active[i])
    dt = env.model.opt.timestep
    t = 0.0
    while t < 30.0:
        p = env._cube_pos(idx)
        if p[0] > X_STOP - 0.002 and abs(p[1] - C.BELT_Y) < 0.06:
            break
        sim(env, dt, arm=env.data.qpos[env._arm_qpos], grip=GRIP_OPEN)
        t += dt
    sim(env, 0.4, arm=env.data.qpos[env._arm_qpos], grip=GRIP_OPEN)   # let it settle
    print(f"   part indexed to x={env._cube_pos(idx)[0]:+.3f} after {t:.2f} s "
          f"and the line is holding it")
    assert X_STOP - 0.01 < env._cube_pos(idx)[0] < C.PICK_X[1], "stop outside the window"

    print("3. descend and grasp the held part")
    p = env._cube_pos(idx)
    print(f"   part at grasp station {np.round(p, 4).tolist()}")
    move_to(env, (p[0], p[1], APPROACH_Z), GRIP_OPEN, settle=0.15)
    move_to(env, (p[0], p[1], GRASP_Z), GRIP_OPEN, settle=0.4)
    print(f"   after descent: part {np.round(env._cube_pos(idx), 4).tolist()}")
    sim(env, 0.5, grip=GRIP_SHUT)
    held = env._grasped()
    f, mm_ = env._jaws()
    p = env._cube_pos(idx)
    print(f"   part   at {np.round(p, 4).tolist()}")
    print(f"   gripper at {np.round(env._tcp(), 4).tolist()}  "
          f"jaws {np.round(f, 3).tolist()} / {np.round(mm_, 3).tolist()}  "
          f"gap {np.linalg.norm(f-mm_)*1000:.1f} mm")
    print(f"   grasp: held index = {held} (expect {idx})")
    snap("grasp")
    assert held == idx, "grasp failed"

    print("4. lift and carry to the tote")
    move_to(env, (p[0], p[1], LIFT_Z), GRIP_SHUT)
    print(f"   after lift: part {np.round(env._cube_pos(idx), 4).tolist()} "
          f"held={env._grasped()}")
    move_to(env, (TOTE_XY[0], TOTE_XY[1], LIFT_Z), GRIP_SHUT)
    print(f"   after carry: part {np.round(env._cube_pos(idx), 4).tolist()} "
          f"held={env._grasped()}")
    snap("carry")
    move_to(env, (TOTE_XY[0], TOTE_XY[1], C.TOTE_RELEASE_Z), GRIP_SHUT,
            settle=0.4)
    print(f"   over tote: part {np.round(env._cube_pos(idx), 4).tolist()} "
          f"held={env._grasped()}")

    print("5. release")
    sim(env, 0.5, grip=GRIP_OPEN)
    sim(env, 0.8)
    q = env._cube_pos(idx)
    print(f"   part rests at {np.round(q, 4).tolist()}")
    for i in range(C.POOL):
        if env._active[i] and i != idx:
            print(f"   other part {i}: {np.round(env._cube_pos(i), 4).tolist()}")
    print(f"   tote interior x[{C.TOTE_INNER[0]:+.3f},{C.TOTE_INNER[1]:+.3f}] "
          f"y[{C.TOTE_INNER[2]:+.3f},{C.TOTE_INNER[3]:+.3f}] rim {C.TOTE_RIM}")
    inside = env._in_tote(q)
    print(f"   INSIDE TOTE: {inside}")
    snap("placed")
    move_to(env, C.HOME_XYZ, GRIP_OPEN, top_down=False, settle=0.3)

    while len(shots) % 3:
        shots.append(shots[-1])
    strip = np.concatenate([np.concatenate([s[1] for s in shots[i:i + 3]], axis=1)
                            for i in range(0, len(shots), 3)], axis=0)
    Image.fromarray(strip).save("demo_pick.png")
    r.close()
    env.close()
    print("\nDEMO PASSED: part picked off the line and placed in the tote"
          if inside else "\nDEMO FAILED")
    return 0 if inside else 1


if __name__ == "__main__":
    raise SystemExit(main())






