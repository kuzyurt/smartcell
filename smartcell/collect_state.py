"""Phase 2: collect the state dataset `ds_0001/` with a randomized expert.

Measured before collection: the env
randomizes NOTHING - `reset(seed)` produced byte-identical episodes for seeds
0 and 7.  So randomization lives HERE, not in the env.  Each episode samples a
vector and applies it through a thin wrapper around the verified expert
(`expert.py` is never edited; the patching below is runtime composition in
this process and is recorded in the manifest).

Randomization vector (stage A - physics; tier-B visual randomization is
Phase 3's job, applied at render time):

  belt_speed  U[0.6, 1.4] x C.BELT_SPEED   re-asserted every physics step.
                                            RUNNING_LINE (below) decides whether
                                            a zero-hold at the indexing stop is
                                            honoured or overridden; the speed is
                                            read back from geom_surfacevel,
                                            never assumed.
  part_yaw    U[-pi, pi]                    part 0 orientation at reset
  jaw_yaw     nearest face normal to +Y,
              + U[-0.15, 0.15] rad          a 32 mm square seen across the jaws
                                            at 45 deg is 45.3 mm wide against a
                                            34 mm open gap, so a *fixed* jaw yaw
                                            would fail ~57% of episodes; the
                                            expert instead squares up to the
                                            part and keeps a small misalignment
  aim_xy      N(0, 2 mm), clip 5 mm         one systematic offset per episode,
                                            applied to every Cartesian target
  hover_dz    U[-20, +20] mm                approach height
  grip_shut   0.020 + U[-1, +3] mm          closing depth

Every episode logs the raw applied ctrl at the physics rate (5 ms), which makes
Phase 3 replay exact, plus the normalized [-1,1] policy action.  Failed
episodes are recorded in the manifest and their npz is not written.

Workers resumable via the manifest: an attempt id already present is skipped.

Run:  python collect_state.py --smoke          1 worker, 3 episodes, verified
      python collect_state.py                  4 workers, 300 episodes
      python collect_state.py --target 3000    Phase 6 scale
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time

# the collector is the one tool that MUST host multiprocessing.spawn
# workers (one MuJoCo env each), and a child cannot be created
# inside the job object, so it takes the cooperative budget.
import os as _os
_os.environ.setdefault("SMARTCELL_RAM_CAP_MODE", "cooperative")
import ramcap          # 5 GB cap, see ramcap.py
import numpy as np
import mujoco

try:
    import torch
except Exception:                                         # pragma: no cover
    torch = None

import build_cell as C
import conveyor_env
import demo_pick
import expert as E
from conveyor_env import (ConveyorPickEnv, set_belt_speed, GRIP_OPEN,
                          GRIP_SHUT, X_STOP, HOME)
from demo_pick import APPROACH_Z
import current

# Per-episode indexing-stop position.  The drive holds a part once its centre
# passes X_STOP, so the resting x is a deterministic function of the belt speed
# and the stop threshold - one fixed number for every episode of ds_0001, which
# is exactly the kind of constant a BC policy memorises instead of using the
# image.  Every episode now samples a small offset and the collector MOVES the
# threshold (conveyor_env.X_STOP is read as a module global at every step, so
# this is a runtime patch, not an edit of the verified env).
X_STOP_NOMINAL = X_STOP
STOP_VAR_M = 0.008              # +-8 mm around 0.100 m; stays inside PICK_X


def apply_stop(vec):
    """Install this episode's indexing stop; returns the threshold used."""
    stop = X_STOP_NOMINAL + float(vec.get("stop_dx", 0.0))
    conveyor_env.X_STOP = stop
    return stop


# Per-episode tote position.  The tote sat at ONE spot in every demo of
# ds_0001/0002/0003, so the policy could carry to a memorised pixel instead of
# looking for the box.  Moving the box forces the policy to use its camera.  Per EPISODE, not per run: one fixed spot per run still lets the
# policy memorise it.  Same runtime-patch pattern as apply_stop:
#   * the nine tote geoms are named WORLD geoms (mjbox.box sets geom.name), so
#     model.geom_pos + mj_forward moves the box;
#   * conveyor_env._in_tote and render_dataset.window_rows read
#     build_cell.TOTE_INNER / TOTE_XY as module attributes AT CALL TIME, so
#     shifting those constants moves the success check with the box.
# demo_pick.TOTE_XY is an import-time copy and demo_pick.py must not be edited:
# the expert's drop waypoint is shifted in _patched_move_to instead.
TOTE_HOME_XY = np.array(C.TOTE_XY, dtype=float)
TOTE_HOME_X = tuple(C.TOTE_X)      # nominals: apply_tote must never compound
TOTE_HOME_Y = tuple(C.TOTE_Y)
# Envelope measured on a 15-offset grid plus a boundary sweep:
# x is free to +-70 mm (drop err 8-21 mm everywhere), +y is walled by the
# arm's reachability near the robot base - dy >= +30 mm sends the IK into its
# reseed-degenerate branch (solves 80-95 vs 128+, 3 vs 5 reseeds) and the part
# is released from a wrong pose (drop err 59-290 mm).  dy <= +20 mm is clean
# (5.0-5.3 mm).  Asymmetric by measurement, not by taste.
TOTE_VAR_X_M = 0.060
TOTE_VAR_Y_M = (-0.060, 0.020)
TOTE_GEOMS = ("tote_floor",
              "tote_wall_n", "tote_wall_s", "tote_wall_w", "tote_wall_e",
              "tote_lip_n", "tote_lip_s", "tote_lip_w", "tote_lip_e",
              "boxmark_floor")
_TOTE_CACHE = {}


def apply_tote(vec, env=None):
    """Install this episode's tote position; returns the (dx, dy) used."""
    dx = float(vec.get("tote_dx", 0.0))
    dy = float(vec.get("tote_dy", 0.0))
    wall = C.TOTE_WALL
    C.TOTE_X = (TOTE_HOME_X[0] + dx, TOTE_HOME_X[1] + dx)
    C.TOTE_Y = (TOTE_HOME_Y[0] + dy, TOTE_HOME_Y[1] + dy)
    C.TOTE_XY = (float(TOTE_HOME_XY[0] + dx), float(TOTE_HOME_XY[1] + dy))
    C.TOTE_INNER = (C.TOTE_X[0] + wall, C.TOTE_X[1] - wall,
                    C.TOTE_Y[0] + wall, C.TOTE_Y[1] - wall)
    conveyor_env.TOTE_C = np.array(C.TOTE_XY)      # obs-only, kept consistent
    if env is not None:
        m, d = env.model, env.data
        ids = _TOTE_CACHE.get("ids")
        if ids is None:
            ids = np.array([int(m.geom(n).id) for n in TOTE_GEOMS])
            assert len(ids) == len(TOTE_GEOMS), "tote geom names not found"
            _TOTE_CACHE["ids"] = ids
            _TOTE_CACHE["home"] = m.geom_pos[ids].copy()
        m.geom_pos[ids] = _TOTE_CACHE["home"] + np.array([dx, dy, 0.0])
        mujoco.mj_forward(m, d)
    return dx, dy

SOURCES = ["collect_state.py", "expert.py", "conveyor_env.py",
           "arm_ik.py", "demo_pick.py", "build_cell.py"]
BASE_SEED = 100000          # eval uses 10000-10049; keep collection disjoint
SEED_SALT = 0x9E3779B9


def code_hash():
    h = hashlib.sha256()
    for f in SOURCES:
        h.update(open(f, "rb").read())
    return h.hexdigest()[:12]


# --------------------------------------------------------------- wrapper env
class RandEnv:
    """Delegates everything to the real env, applies the vector after reset.

    `expert.sim` calls `env._conveyor_logic()` every step; the delegated call
    is wrapped so the sampled belt speed is re-asserted after the env's own
    logic (which resets it to stock whenever no part waits at the stop).
    """

    def __init__(self, real):
        self.real = real
        self.vec = None

    def reset(self, *, seed=None, options=None):
        obs = self.real.reset(seed=seed)
        v = self.vec
        m, d = self.real.model, self.real.data
        a = self.real._cube_qpos[0]
        d.qpos[a + 3:a + 7] = [np.cos(v["part_yaw"] / 2), 0.0, 0.0,
                               np.sin(v["part_yaw"] / 2)]
        mujoco.mj_forward(m, d)
        set_belt_speed(m, v["belt_speed"])
        self.row0 = dict(t=0.0, qpos=d.qpos[self.real._arm_qpos].copy(),
                         grip=d.qpos[self.real._grip_qpos],
                         tcp=self.real._tcp(),
                         cubes=np.stack([self.real._cube_pos(i)
                                         if self.real._active[i] else np.zeros(3)
                                         for i in range(C.POOL)]),
                         active=np.array(self.real._active, dtype=np.int8))
        return obs

    def _logic_and_belt(self):
        self.real._conveyor_logic()
        m = self.real.model
        belt = int(m.geom("belt").id)
        if RUNNING_LINE:
            # The line NEVER stops (user: "the assembly line shouldnt stop. it
            # should always go").  conveyor_env._conveyor_logic zeroes
            # geom_surfacevel while a part waits at the indexing stop; we
            # overwrite that request every physics step.  conveyor_env.py is
            # untouched - this wrapper is where the sampled speed has always
            # been re-asserted, so it is the whole change, and evaluate.py
            # gets it too because it drives the same RandEnv.
            set_belt_speed(m, self.vec["belt_speed"])
        elif abs(float(m.geom_surfacevel[belt][0])) > 1e-9:   # 0 = hold: keep it
            set_belt_speed(m, self.vec["belt_speed"])

    def __getattr__(self, name):
        if name == "_conveyor_logic":
            return self._logic_and_belt
        return getattr(self.real, name)

    def __setattr__(self, name, value):
        # writes must land on the real env: `env._sim_t += dt` on a plain
        # wrapper would SHADOW the attribute and freeze the real env's clock
        # at 0, silently disabling spawn timing in _conveyor_logic.
        if name in ("real", "vec", "row0"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.real, name, value)


# ---------------------------------------------------- expert patch (runtime)
CTX = {}


def _face_yaw(part_yaw, rng):
    """Face-normal angle nearest world +Y (mod pi), plus jitter.

    Returns (angle, unit 3-vector).  ArmIK aligns the jaw axis (site +X) to
    its `yaw` 3-vector, and the jaws must close ALONG a face normal of the
    square: for part_yaw = 0 that is exactly the demo's (0, 1, 0).

    Measured: EXACT alignment places 16/16 across all yaws;
    +-0.03 rad of jaw jitter drops to ~75-83%, all in mid-carry slip.  The
    sampler therefore keeps alignment exact and lets part_yaw vary which
    face is grasped - the diversity channel is the part, not the jaws.
    """
    cands = [(part_yaw + k * np.pi / 2) % np.pi for k in range(2)]
    base = min(cands, key=lambda a: abs(a - np.pi / 2))
    ang = base
    return ang, np.array([np.cos(ang), np.sin(ang), 0.0])


def sample_vector(seed):
    """The randomization vector, inside the measured-stable envelope.

    Envelope choices, all measured as full-episode placed rate:
      jaw-to-face alignment EXACT  (jitter 0.03 rad -> 75-83% placed)
      aim jitter sigma 1 mm        (2 mm -> 92%, 5 mm -> 67%, all carry slip)
      grip_shut 0.020 +- 1 mm      (smoke's +3 mm looser squeeze co-sinned)
    Free channels remain: part yaw full circle, belt speed 0.6-1.4x,
    hover height +-20 mm.  Visual randomization and latency are Phase 3's
    noise budget; the expert's own aims stay clean for BC labels.

    VARIANCE UPGRADE (ds_0011+, Oct 2026): the old vector memorised timing.
    belt fixed + lead fixed 0.9s + wait pose fixed = every episode breaks the
    wait at the same belt position and drives the same Cartesian line. A BC
    policy can solve training by memorising t->action instead of looking at
    the cube. New channels below are INDEPENDENT (research: independently
    varied factors generalise better than confounded ones at the same budget;
    env/object diversity outranks demo count; 50 demos per env-object pair
    saturates). All bounds are physically possible and validated by the
    expert's own success check (failed episodes DROP, never silently kept):
      wait pose +-20 mrad/joint (~1.1deg): camera still sees the lane
        (wait pose looks 4.4deg off stop vs 30.5deg half-fov; 1deg jitter safe)
      belt U[0.068,0.092] around 0.08 nominal (+-15%): window 1.5-2.1s,
        still inside the 0.68-0.86s approach + chase + 0.8s close budget.
        Old 0.6-1.4x reached 0.13 m/s where the arm drops boxes.
      lead U[0.7,1.1]s: pick time varies by ~0.4s (~32mm of belt travel),
        always arriving early (chase re-aims; late is unrecoverable).
      visibility gate: break only when part within 120mm before stop
        (|y-BELT_Y|<60mm), so the cube is in the wrist frame when the arm
        moves. An invisible pick is a DROP, not a demo.
      path bow U[-15,15]mm x/y on the approach via-point + lift U[-20,20]mm:
        same endpoints, different line; clears tote rim (0.715) at all lifts.
      settle scale U[0.8,1.2] on non-critical settles: arm speed varies ~20%
        without touching the 0.5+0.3s close (jaws need the time to shut).
      chase sensing N(0,3mm): blindfold-lite. The expert aims at a noisy
        read (like a vision policy must), then corrects. 3mm << 17mm capture
        band, so it still succeeds while demonstrating corrections.
    """
    rng = np.random.default_rng((seed ^ SEED_SALT) & 0xFFFFFFFFFFFFFFFF)
    part_yaw = float(rng.uniform(-np.pi, np.pi))
    jy, jy_vec = _face_yaw(part_yaw, rng)
    if BELT_NARROW_VAR:
        belt = float(rng.uniform(BELT_VAR_LO_MPS, BELT_VAR_HI_MPS))
    else:
        belt = (float(C.BELT_SPEED) if BELT_FIXED
                else float(rng.uniform(0.6, 1.4) * C.BELT_SPEED))
    vec = dict(
        belt_speed=belt,
        part_yaw=part_yaw,
        jaw_yaw=jy,
        jaw_yaw_vec=jy_vec,
        aim_xy=np.clip(rng.normal(0.0, 0.001, 2), -0.002, 0.002),
        hover_dz=float(rng.uniform(-0.02, 0.02)),
        grip_shut=0.020 + float(rng.uniform(-0.001, 0.001)),
        stop_dx=float(rng.uniform(-STOP_VAR_M, STOP_VAR_M)),
        grasp_dz=GRASP_DZ_M,
        # drawn LAST so the rest of the stream matches ds_0001/2/3 exactly
        tote_dx=float(rng.uniform(-TOTE_VAR_X_M, TOTE_VAR_X_M)),
        tote_dy=float(rng.uniform(*TOTE_VAR_Y_M)),
    )
    # variance upgrade channels (drawn after the legacy stream)
    vec["wait_lead_s"] = float(rng.uniform(WAIT_LEAD_LO_S, WAIT_LEAD_HI_S))
    vec["settle_scale"] = float(rng.uniform(SETTLE_LO, SETTLE_HI))
    vec["lift_dz"] = float(rng.uniform(LIFT_DZ_LO_M, LIFT_DZ_HI_M))
    vec["bow_xy"] = [float(x) for x in rng.uniform(BOW_LO_M, BOW_HI_M, 2)]
    # wait pose jitter applied by worker_main (needs base pose); stored as
    # the ACTUAL start pose so replayers stay exact (install_wait_pose reads it)
    return vec


# --- variance upgrade constants (ds_0011+) ---
BELT_NARROW_VAR = True
"""True: per-episode belt U[0.068,0.092] around the 0.08 nominal (new way).

Narrow and sensible: +-15% keeps every episode inside the arm's proven
budget (measured: 0.12 misses 2/24, 0.14 misses 3/24 with 8 catches).
The old U[0.6,1.4]x reached 0.13 at nominal 0.08 - past where the arm drops.
False restores BELT_FIXED / --belt-jitter behaviour for exact reproduction.
Set by --belt-narrow-var / --belt-fixed (re-applied in workers: spawn)."""
BELT_VAR_LO_MPS = 0.068
BELT_VAR_HI_MPS = 0.092
WAIT_LEAD_LO_S = 0.7
WAIT_LEAD_HI_S = 1.1
WAIT_JITTER_RAD = 0.020
VIS_X_M = 0.120
SETTLE_LO, SETTLE_HI = 0.8, 1.2
LIFT_DZ_LO_M, LIFT_DZ_HI_M = -0.020, 0.020
BOW_LO_M, BOW_HI_M = -0.015, 0.015
CHASE_NOISE_M = 0.003


def sample_wait_joints(base, rng):
    """Per-episode wait pose: base +-20mrad/joint, clipped on use.

    Slight by design (user): a bigger move would put the pick out of reach.
    The clip to actuator range happens in episode_stable_wait/install_wait_pose
    (range comes from the model, not from here). Returns list of 5 floats.
    """
    base = np.asarray(base if base is not None else current.WAIT_JOINTS,
                      dtype=float)
    j = base + rng.uniform(-WAIT_JITTER_RAD, WAIT_JITTER_RAD, size=base.shape)
    return [float(x) for x in j]


def part_visible_from_wait(p, stop):
    """Visibility gate: cube must be in the wrist frame before the arm moves.

    Exact frustum needs the Mount at the jittered wait pose; the conservative
    proxy is: within 120mm before the stop and |y-lane|<60mm. The wait pose
    looks 4.4deg off the stop (30.5deg half-fov), so anything in this window
    is in frame; anything behind it may not be. Never guess: if in doubt the
    episode waits (or DROPs on timeout) instead of picking blind.
    """
    return bool(p[0] > stop - VIS_X_M and abs(p[1] - C.BELT_Y) < 0.06)


BELT_FIXED = False
"""True: every episode runs the belt at exactly C.BELT_SPEED, no U[0.6, 1.4].

The speed limit is only meaningful at one speed - "the robot takes every box
at 0.06 m/s" has to be a statement about 0.06 m/s, not about a distribution
whose slow tail is trivially easy and whose fast tail is not.  Set by
--belt-fixed; re-applied in worker_main for the same spawn reason as
WAIT_JOINTS.
"""


GRASP_CATCH_M = 0.004
"""How close the chase has to get, in metres, before it stops correcting.

The jaws are 66 mm open and the part is 32 mm, so the capture band is +-17 mm
(GRASP_SHUT=17 mm is where the servo stalls).  4 mm leaves the chase a wide
margin inside that band while keeping the final correction short enough that
the part has not moved again by the time it lands.
"""

WAIT_LEAD_S = 0.9
"""Break the wait this long BEFORE the part reaches the station.

Measured: the approach leg - wait pose to hover above the
part, IK plus its 0.15 s settle - took 0.68 s, and 0.86 s on the seeds whose
episodes went on to fail.  The wait used to break the instant the part passed
the stop, which makes "late" - the arm arriving after the part has left the
window - the single largest failure mode, and a speed-INDEPENDENT one: 1 of 12
at 0.07, 0.09 and 0.14 m/s alike.

0.9 s is the measured worst leg plus margin.  At 0.12 m/s that is 108 mm of
lead, so the arm is already moving while the part is still 108 mm short of the
station.  Arriving EARLY is harmless - the chase re-aims at the part's actual
position - and arriving late is unrecoverable, so the bias is deliberately all
in one direction.
"""

TOTE_LOOK_S = 0.2
TOTE_LOOK_H_M = 0.06
TOTE_LOOK = True
"""Tote-look legs: verify-hold above the tote before every descent.

The wrist camera barely sees the tote during the carry (E3: in-frame
14-17% of carry grids, median 0px) - a policy can then only place from a
memorised tote position, which the +-60mm tote variance forbids. So the
expert demonstrably LOOKS (when TOTE_LOOK is True): it descends to 60mm
above the release point (part bottom ~4mm clear of the rim), centres the
box mouth in frame with near-field jogs (see _center_tote_in_view),
holds TOTE_LOOK_S with the part in the jaws, then descends the final
60mm. The hold is 0.2s = 5 frames ON PURPOSE: the policy's 8-frame stack
spans 280ms, so a longer hold saturates the stack with identical frames
and the resume time becomes unobservable - measured, a 0.8s hold taught
the policy to hover over the tote forever (6/10 nominal picks held to
the 20s horizon, 0% place). With a 0.2s beat the whole hold fits inside
one stack window and the descend decision stays learnable. Fixed (not
settle-scaled): every episode shows the same verify pattern, recorded
in the line record (look_hover_s) and stats.
TOTE_LOOK=False for dagger_collect: the command-walk applies drift
offsets inside E.sim AND times its windows in seconds, so it fights the
centering (measured residuals to (+999,+999) depth-negative, fail rate
15/64 with a not-in-tote) and misaligns walk phases. Dagger rounds stay
look-free (like d1-d14); the look pattern comes from the dedicated look
dataset only. Read at call time by episode_stable_wait."""

CATCH_ITERS = 8
"""How many times the chase may re-aim before the jaws close anyway.

Measured: a converged chase costs 0.08-0.22 s per
correction, and with only 3 the episode miss rate was 2/24 at 0.12 m/s and
3/24 at 0.14 - every miss "late": the arm arrived after the part had passed
the window and ran out of corrections while catching up.  The correction is
the cheapest thing available, a short IK move at grasp height with the jaws
open, and the loop exits on the first check that passes, so extra iterations
cost nothing when the chase has already converged.
"""


def line_config():
    """How the line was running, recorded in every manifest row.

    A dataset is only interpretable if it says whether the belt stopped for
    the robot or not, at what speed, and with the lead and catch constants the
    pick depended on.  Without this, two datasets collected with the same code
    hash could still differ in exactly the variable that decides whether the
    demo is learnable.
    """
    if BELT_NARROW_VAR:
        jitter = [BELT_VAR_LO_MPS, BELT_VAR_HI_MPS]
    else:
        jitter = (None if BELT_FIXED else [0.6, 1.4])
    return dict(running=bool(RUNNING_LINE),
                belt_nominal=round(float(C.BELT_SPEED), 4),
                belt_jitter=jitter,
                belt_narrow_var=bool(BELT_NARROW_VAR),
                spawn_every_s=round(float(conveyor_env.SPAWN_EVERY), 3),
                wait_lead_s=(f"U[{WAIT_LEAD_LO_S},{WAIT_LEAD_HI_S}] per-ep"
                             if RUNNING_LINE else None),
                wait_lead_default=WAIT_LEAD_S,
                catch_iters=(CATCH_ITERS if RUNNING_LINE else None),
                catch_band_mm=(round(GRASP_CATCH_M * 1000.0, 2)
                               if RUNNING_LINE else None),
                wait_jitter_rad=WAIT_JITTER_RAD,
                look=bool(TOTE_LOOK),
                look_hover_s=(round(float(TOTE_LOOK_S), 2)
                              if TOTE_LOOK else None),
                vis_x_m=VIS_X_M,
                settle_scale=f"U[{SETTLE_LO},{SETTLE_HI}]",
                lift_dz_m=[LIFT_DZ_LO_M, LIFT_DZ_HI_M],
                bow_m=[BOW_LO_M, BOW_HI_M],
                chase_noise_m=CHASE_NOISE_M)


SNAP = None


def snap(env, idx, tag):
    """Optional geometry trace, enabled through SNAP."""
    if SNAP is not None:
        SNAP(env, idx, tag)


def _belt_v(env):
    """The belt's actual commanded surface speed, m/s, read not assumed.

    Read from geom_surfacevel rather than from C.BELT_SPEED or the episode
    vector: RUNNING_LINE re-asserts the sampled speed every step, and the
    chase lead is only correct if it uses the speed the belt is really running.
    """
    g = int(env.model.geom("belt").id)
    return float(env.model.geom_surfacevel[g][0])


RUNNING_LINE = False
"""True: the belt never stops, so the part is a MOVING target at pick time.

The env's own indexing stop (conveyor_env._conveyor_logic zeroing
geom_surfacevel while a part waits at X_STOP) is what made the old demos
stationary-target demos.  The target setup is a line that always runs, as a
real conveyor does not stop for the robot, at a speed the robot can still take
every box off.  RUNNING_LINE makes RandEnv re-assert the sampled speed every
physics step, which overrides the hold; the pick waypoints then have to LEAD
the part (see episode_stable_wait step 3).

Set from the command line by --running-line, and re-applied inside worker_main
because the worker pool is spawn and module globals do not survive it.
"""


WAIT_JOINTS = None
"""Optional joint-space WAIT pose, overriding the Cartesian hover below.

None (the default, and what ds_0001..ds_0006 used) parks the arm OVER THE STOP,
because the wrist camera aims at the grasp site and that is what puts the lane
in shot while the arm waits.

The joint-space wait pose, [-1.440, -1.465, 0.300, 1.657,
-1.449] (ds_0007 onwards), holds for 40.7% of the episode and was checked for
lane visibility WITH THE PHYSICS STEPPED: the angle from the wrist camera's
optical axis to the stop is 4.4 deg, to the grasp site 21.4 deg, against a
30.5 deg half-fov - the camera looks straight down the lane.  (An earlier
measurement here sampled with dt=0, so the arm never moved and it reported the
pose as blind; that number was wrong and the pose is good.)

When this is set, q_start is the wait pose, not conveyor_env.HOME, because the
user requires the episode to END where it STARTS.
"""


_LOOK_MOUNT = [None, None]  # (Mount, id(model)): one env per worker


def get_mount(m):
    """The ONE wrist-cam Mount for a model (shared, idempotent).

    V.Mount() world-parents the camera as a side effect, so a second
    V.Mount(m) on the same model trips check_mount and dies - which is
    what killed dagger_collect (it builds its own Mount at startup, then
    the look centering built another). Every caller that needs the wrist
    pose on a shared model must come through here.
    """
    import cam_variant as V
    if _LOOK_MOUNT[0] is None or _LOOK_MOUNT[1] != id(m):
        _LOOK_MOUNT[0] = V.Mount(m)
        _LOOK_MOUNT[1] = id(m)
    return _LOOK_MOUNT[0]


def _center_tote_in_view(env, stats=None, log=False):
    """Centre the tote mouth in the wrist frame by visual servoing.

    Returns the final (offx, offy) in frame pixels. Max 3 iterations,
    15px gate, gain 0.7, 80mm clamp. Called at LOW look height (caller
    descends first). The site->frame map here is dominated by wrist
    REORIENTATION (measured: a site move landing exactly as commanded
    still swung framing 20-60px off the parallax prediction), so the
    local 2x3 Jacobian is estimated by finite differences through the
    real IK + forward kinematics each iteration and the site steps along
    -J+.off. Moves land exactly (measured: commanded vs landed agree to
    0.1mm), so the only error is linearization, re-estimated per step.
    All moves RAW (_orig_move_to, exact site placement - the patch's
    ~14mm jaw shift would fight the correction), logged, replay-exact.
    Deterministic (no RNG). Part in free air throughout (mouth 200x180
    vs 32mm part).
    """
    r = env.real
    m, d = r.model, r.data
    mount = get_mount(m)
    cid = int(m.camera("wrist_cam").id)
    fovy = float(m.cam_fovy[cid])
    tan_half = float(np.tan(np.radians(fovy / 2.0)))
    tc = np.array([C.TOTE_XY[0], C.TOTE_XY[1], C.TABLE_TOP + 0.045])
    def read_off():
        mujoco.mj_forward(m, d)
        mount.update(d)
        cp = np.array(m.cam_pos[cid], float)
        cm = np.array(d.cam_xmat[cid], float).reshape(3, 3)
        rel = tc - cp
        pcam = cm.T @ rel
        depth = -float(pcam[2])
        if depth <= 0.003:
            return (999.0, 999.0)
        return (float(pcam[0] / depth / tan_half) * 64.0,
                -float(pcam[1] / depth / tan_half) * 64.0)

    yawv = CTX.get("jaw_yaw_vec", (0.0, 1.0, 0.0))

    def proj_at(q):
        d.qpos[r._arm_qpos] = q
        mujoco.mj_forward(m, d)
        mount.update(d)
        cp = np.array(m.cam_pos[cid], float)
        cm = np.array(d.cam_xmat[cid], float).reshape(3, 3)
        rel = tc - cp
        pcam = cm.T @ rel
        depth = -float(pcam[2])
        if depth <= 0.003:
            return (999.0, 999.0)
        return (float(pcam[0] / depth / tan_half) * 64.0,
                -float(pcam[1] / depth / tan_half) * 64.0)

    off = read_off()
    for _ in range(3):
        if abs(off[0]) <= 15.0 and abs(off[1]) <= 15.0:
            break
        # image-based visual servoing: the site->frame map here is
        # dominated by wrist REORIENTATION (moving the site 27mm can swing
        # framing 20-60px the "wrong" way - measured), so no fixed
        # parallax model can work. Instead estimate the local 2x3 Jacobian
        # by finite differences through the real IK + forward kinematics,
        # then step site += -J+.off. Moves land exactly (measured: commanded
        # vs landed site agree to 0.1mm), so the only error is
        # linearization - re-linearized every iteration. Deterministic.
        q_here = d.qpos[r._arm_qpos].copy()
        site = np.array(d.site_xpos[r._grasp_site], float)
        J = np.zeros((2, 3))
        h = 0.005
        for a in range(3):
            dt = np.zeros(3)
            dt[a] = h
            q, _, _ = r.ik.solve(site + dt, q0=q_here, top_down=True,
                                 yaw=yawv, outer=2, iters=12,
                                 restarts=0)[:3]
            o = np.array(proj_at(q))
            J[:, a] = (o - np.array(off)) / h
        d.qpos[r._arm_qpos] = q_here
        mujoco.mj_forward(m, d)
        step, _, _, _ = np.linalg.lstsq(J, -0.7 * np.array(off), rcond=None)
        n = float(np.linalg.norm(step))
        if n > 0.080:
            step = step * (0.080 / n)
        site = np.array(d.site_xpos[r._grasp_site], float)
        _orig_move_to(env, site + step, GRIP_SHUT, settle=0.2, log=False,
                      cfg=E.FAST, stats=stats)
        off = read_off()
    if log or max(abs(off[0]), abs(off[1])) > 40.0:
        print(f"  tote look residual ({off[0]:+.0f}, {off[1]:+.0f})px",
              flush=True)
    return off


def episode_stable_wait(env, seed, cfg=E.FAST, log=False, hook=None, stats=None):
    # The pose this episode must END in: conveyor_env.HOME, which is exactly
    # what ConveyorPickEnv.reset() writes into qpos.  The return-to-home at the
    # end of the episode must command THESE joint angles, not an IK solve to a
    # Cartesian point: move_to(..., top_down=False) leaves the wrist
    # unconstrained, so it reaches C.HOME_XYZ in whatever branch the solver
    # prefers.  Measured on ds_0006 ep 0-2 that left the arm up to 115.3 deg
    # from the start pose on a single joint (wrist_roll ended at 2.3054 against
    # 0.4084) - the tip was in the right place and the arm was folded into a
    # different pose: visibly not the position it started from.
    #
    # It is NOT captured from data.qpos at entry: the collector reuses one env
    # across episodes and never calls reset() per episode (worker_main only
    # re-samples env.vec), so qpos at entry is the PREVIOUS episode's end pose
    # - measured [0,0,0,0,0] on episode 0 of a fresh worker - and returning to
    # that would send the arm to all-zero, which is not home either.
    # ...unless an explicit joint-space WAIT pose is given, in which case
    # THAT is the pose the episode starts in and must also end in (start and
    # end position must be the same).  HOME
    # is what reset() writes, so it is only correct when the episode really
    # does begin at home - which is not the case with --wait-joints.
    # per-episode start pose: the jittered wait_joints travel in the vector
    # (worker_main writes the ACTUAL jittered pose there), so replayers stay
    # exact. Falls back to the module WAIT_JOINTS / HOME for old vectors.
    _wj = (env.vec.get("wait_joints") if isinstance(env.vec, dict) else None)
    if _wj is not None:
        q_start = np.asarray(_wj, float).copy()
    else:
        q_start = (np.asarray(WAIT_JOINTS, float).copy()
                   if WAIT_JOINTS is not None else np.asarray(HOME, float).copy())
    """expert.episode with a STABLE belt wait.

    demo_pick's wait uses arm=qpos (feedback braking); that is unstable under
    this actuator model - the arm flails for seconds at every episode start
    (measured: pan -0.45 -> -1.06 rad, elbow 0.78 -> -0.19).  The expert
    re-solves IK from anywhere and does not care; BC imitates the flail and
    the whole window becomes garbage for learning.  Here the arm is driven to
    the hover pose FIRST (IK, as the expert does), then HELD there with a
    constant ctrl while the belt runs - a deterministic wait the policy can
    actually learn.  expert.py itself is NOT modified.
    """
    from demo_pick import (move_to as _noop,)  # noqa: F401  (import guard)
    import demo_pick as D
    env._arm_lo = env.model.actuator_ctrlrange[env._arm_aid, 0]
    env._arm_hi = env.model.actuator_ctrlrange[env._arm_aid, 1]
    q_start = np.clip(q_start, env._arm_lo, env._arm_hi)
    stop = apply_stop(env.vec)
    apply_tote(env.vec, env)
    # per-episode variance knobs (defaults = old fixed behaviour for old vecs)
    _lead = float(env.vec.get("wait_lead_s", WAIT_LEAD_S))
    _settle = float(env.vec.get("settle_scale", 1.0))
    _lift_dz = float(env.vec.get("lift_dz", 0.0))
    _bow = env.vec.get("bow_xy", [0.0, 0.0])
    try:
        _bow = [float(_bow[0]), float(_bow[1])]
    except Exception:
        _bow = [0.0, 0.0]
    _chase_rng = np.random.default_rng((int(seed) ^ 0xC4A5E) & 0xFFFFFFFF)
    env.reset(seed=seed)
    # The episode starts at the wait pose, not at HOME.  reset() writes HOME
    # into qpos, so the recorded t=0 used to be HOME and the wait pose only
    # arrived after a 2 s drive.  The episode has to begin and end at the wait
    # pose, and the first recorded frame has to show that pose.  The helper
    # also re-snaps row0, because states_array writes row0 as the t=0 row.
    install_wait_pose(env)
    if stats is None:
        stats = dict(ik_s=0.0, solves=0, sim_s=0.0, steps=0, reseeds=0)
    sim_ = env._sim_t

    # 1) go to hover immediately (IK path, one full search: reseed=True)
    #
    # Park OVER THE STOP, not at the old D.HOVER_X = 0.02 "window entry" point.
    # The part comes to rest at x = stop (~0.10), so hovering at 0.02 left it
    # ~80 mm off the optical axis.  The wrist camera aims at the GRASP SITE, so
    # during the wait it was staring at empty table with the 32 mm part reduced
    # to a ~3 px speck in the 96x96 model input.  Parking over the stop is what
    # makes the camera look at the lane it is about to pick from.
    if _wj is not None or WAIT_JOINTS is not None:
        # drive straight to the (per-episode jittered) joint angles and hold;
        # no IK, so whatever the camera can see from there is what gets recorded
        q_w = np.clip(np.asarray(q_start, float), env._arm_lo, env._arm_hi)
        E.sim(env, 2.0, arm=q_w, grip=GRIP_OPEN, hook=hook)
        if log:
            print(f"  wait at joint pose {np.round(q_w, 3).tolist()}, "
                  f"site {np.round(env.data.site_xpos[env._grasp_site], 3).tolist()}")
    else:
        E.move_to(env, (float(stop), C.BELT_Y, D.APPROACH_Z), GRIP_OPEN,
                  settle=0.1, log=log, cfg=cfg, stats=stats, reseed=True)
    idx = next((i for i in range(C.POOL) if env._active[i]), None)
    if idx is None:
        return None, False, stats

    # 2) STABLE wait at hover: constant ctrl, not arm=qpos braking
    dt = env.model.opt.timestep
    q_hold = env.data.qpos[env._arm_qpos].copy()
    t = 0.0
    # the trigger the wait breaks on: the station itself on a stopping line,
    # the station pulled back by the distance the part covers during the
    # approach on a running one (per-episode wait_lead_s: pick-time variance).
    # Visibility gate: the part must ALSO be in the wrist frame
    # (within VIS_X_M before stop). An invisible pick is a DROP, not a demo.
    trigger = stop - (_belt_v(env) * _lead if RUNNING_LINE else 0.002)
    stats["trigger_x"] = round(float(trigger), 4)
    stats["wait_lead_s"] = round(float(_lead), 3)
    while t < 30.0:
        p = env._cube_pos(idx)
        if p[0] > trigger and abs(p[1] - C.BELT_Y) < 0.06:
            if (not RUNNING_LINE) or part_visible_from_wait(p, stop):
                break
        E.sim(env, dt, arm=q_hold, grip=GRIP_OPEN, hook=hook)
        t += dt
    E.sim(env, 0.4 if not RUNNING_LINE else 0.05, arm=q_hold, grip=GRIP_OPEN,
          hook=hook)
    if RUNNING_LINE:
        # The line runs, so the part is somewhere PAST the stop when the arm
        # starts down and there is no window to be outside of any more: it is
        # a target the chase tracks.  The only thing that still ends the
        # episode is running out of belt - BELT_END_X is 635 mm past the stop,
        # and the remaining legs (approach, descent, chase, 0.8 s of closing)
        # need 0.2 m of it at 0.14 m/s.
        if not env._cube_pos(idx)[0] < C.BELT_END_X - 0.20:
            return idx, False, stats
    elif not (stop - 0.01 < env._cube_pos(idx)[0] < C.PICK_X[1]):
        return idx, False, stats

    # 3) the rest is exactly expert.episode's approach/grasp/carry/place
    # Path variance: bowed via-point (same endpoints, different line).
    # Physically possible: no barriers (BARRIERS=False), open lane, small bow.
    p = env._cube_pos(idx)
    if abs(_bow[0]) + abs(_bow[1]) > 1e-9:
        E.move_to(env, (p[0] + _bow[0], p[1] + _bow[1], D.APPROACH_Z),
                  GRIP_OPEN, settle=0.05 * _settle,
                  log=False, cfg=cfg, stats=stats)
        p = env._cube_pos(idx)  # re-read: the part moved during the bow leg
    E.move_to(env, (p[0], p[1], D.APPROACH_Z), GRIP_OPEN,
              settle=0.15 * _settle, log=log, cfg=cfg, stats=stats)
    if RUNNING_LINE:
        # THE PART IS MOVING.  The descent and a corrective chase, each aimed
        # at where the part will be when that leg finishes and each re-reading
        # the part first, so the aim does NOT depend on predicting how long the
        # IK takes:
        #
        #   1. hover above where the part is now (above, unchanged);
        #   2. descend to grasp height on the part's CURRENT x;
        #   3. CHASE along the belt at grasp height, to the part's current x
        #      plus a quarter of the closing travel - the last correction.
        #
        # Why a quarter-lead and not "aim exactly at it": the fingers close
        # symmetrically onto a part that is still being carried forward, and
        # the capture is ONE-SIDED.  Jaws closing BEHIND the part push it into
        # the gap and it is pinched; jaws closing AHEAD of it push it away down
        # the belt and it is lost.  So the aim is deliberately a little behind
        # the part's position at the end of the close, which is a quarter of
        # the closing travel - that lands the part mid-close in the tolerance
        # band instead of at its edge.
        #
        # The chase at grasp height is safe while the jaws are OPEN: they are
        # 66 mm apart and the widest a 32 mm part presents (yaw randomised) is
        # 45.3 mm, so the fingers sweep past it without touching.
        #
        # The close stays AFTER the chase.  Closing during it - the jaws can
        # only stall on the part once inside the +-17 mm capture band, so it
        # sounds strictly safer - was measured WORSE: 18/24 at 0.12 m/s
        # against 21/24 with the close after, same seeds.
        # A closed jaw then carries the part with every further chase
        # correction while the belt drags it back, and that fight costs more
        # than the capture wins.
        v = _belt_v(env)
        snap(env, idx, "wait-break")
        p = env._cube_pos(idx)
        E.move_to(env, (p[0], p[1], D.GRASP_Z), GRIP_OPEN, settle=0.0,
                  log=log, cfg=cfg, stats=stats)
        snap(env, idx, "descended")
        for _ in range(CATCH_ITERS):
            if not env._active[idx] or env._grasped() == idx:
                break
            p = env._cube_pos(idx)
            site = float(env.data.site_xpos[env._grasp_site][0])
            dx = abs(p[0] - site)
            if dx <= GRASP_CATCH_M:
                break
            # blindfold-lite: aim at a noisy read (vision-like), break on truth.
            # 3mm << 17mm capture band: still succeeds, demonstrates corrections.
            pn = p + np.append(
                _chase_rng.normal(0.0, CHASE_NOISE_M, 2), [0.0])
            E.move_to(env, (pn[0] + v * 0.25, pn[1], D.GRASP_Z), GRIP_OPEN,
                      settle=0.0, log=log, cfg=cfg, stats=stats)
            snap(env, idx, "chase")
    else:
        p = env._cube_pos(idx)
        E.move_to(env, (p[0], p[1], D.GRASP_Z), GRIP_OPEN, settle=0.4, log=log,
                  cfg=cfg, stats=stats)
    snap(env, idx, "before-close")
    E.sim(env, 0.5, grip=GRIP_SHUT, hook=hook)
    snap(env, idx, "close-0.5s")
    E.sim(env, 0.3, grip=GRIP_SHUT, hook=hook)
    snap(env, idx, "close-0.8s")
    held = env._grasped()
    stats["catch_err_mm"] = round(
        abs(float(env.data.site_xpos[env._grasp_site][0])
            - float(env._cube_pos(idx)[0])) * 1000.0, 2)
    if held != idx:
        return idx, False, stats
    p = env._cube_pos(idx)
    _lift = float(D.LIFT_Z + _lift_dz)
    E.move_to(env, (p[0], p[1], _lift), GRIP_SHUT, log=log, cfg=cfg,
              stats=stats)
    # NO flat carry waypoint: the old (TOTE_XY, LIFT_Z) leg + 0.5 s settle
    # taught a hover the policy never leaves (measured on ckpt/policy_0033:
    # trace_0033 15/20 place failures hover 8-57 mm over the tote at carry
    # height, grip shut, hunting +-15 cm in y to the 20 s horizon - the dip
    # transition fires only from the exact static settle state, which the
    # hunting policy never reproduces).  The carry now runs straight on to
    # the look/release height with no static hover frames: diagonal when
    # the look legs run, high transit with settle=0.0 on the dagger path
    # (walk + centering are incompatible, so dagger skips the look).
    # TOTE LOOK: descend to LOW look height (60mm above release, part
    # bottom ~4mm clear of the rim), CENTER the box mouth in frame with
    # near-field jogs, hold a 0.2s beat with the part in the jaws while
    # the wrist camera verifies the box, then descend the final 60mm.
    # Low, because
    # at carry height the wrist orientation points past the box and no
    # +-120mm hover fixes orientation error (measured: best corner still
    # +75px off); near the tote, near-field jogs land faithfully
    # (measured over 103 episodes: median residual [4,6]px, 97% in gate).
    # All legs logged, replay exact; part in free air throughout.
    # Skipped when TOTE_LOOK is False (dagger_collect: the walk fights it).
    _look_z = float(C.TOTE_RELEASE_Z + TOTE_LOOK_H_M)
    if TOTE_LOOK:
        E.move_to(env, (D.TOTE_XY[0], D.TOTE_XY[1], _look_z), GRIP_SHUT,
                  log=False, cfg=cfg, stats=stats, settle=0.0)
        look_off = _center_tote_in_view(env, stats=stats, log=log)
        q_look = env.data.qpos[env._arm_qpos].copy()
        E.sim(env, TOTE_LOOK_S, arm=q_look, grip=GRIP_SHUT, hook=hook)
        stats["look_s"] = round(float(TOTE_LOOK_S), 2)
        stats["look_off_px"] = [round(float(v), 1) for v in look_off]
    else:
        # dagger path: high transit with NO settle pause (a settle=0.5 leg
        # here re-teaches the hover the fix removes: walked hover states
        # labelled "return to hover").  A bend, not a waypoint.
        E.move_to(env, (D.TOTE_XY[0], D.TOTE_XY[1], _lift), GRIP_SHUT,
                  log=log, cfg=cfg, stats=stats, settle=0.0)
    E.move_to(env, (D.TOTE_XY[0], D.TOTE_XY[1], C.TOTE_RELEASE_Z), GRIP_SHUT,
              settle=0.4 * _settle, log=log, cfg=cfg, stats=stats)
    E.sim(env, 0.5, grip=GRIP_OPEN, hook=hook)
    E.sim(env, 0.8, grip=GRIP_OPEN, hook=hook)
    q = env._cube_pos(idx)
    ok = bool(env._in_tote(q))
    # RETURN HOME.  demo_pick does this at the end of its cycle; the collector
    # did not, so every collected episode simply STOPPED at the tote: measured
    # on ds_0005/ep 0, the arm was frozen at the release pose for the last
    # 0.79 s and the rendered window covered only 43.9% of the episode.
    # Combined with a 30 s eval horizon, every frame after the release was
    # extrapolation - which is the "it picks one and then does not know what to
    # do" failure.  This is the only demonstration of the post-task behaviour.
    if WAIT_JOINTS is None:
        # Lead-in along the path demo_pick uses.  Skipped when the episode
        # starts at a joint-space WAIT pose: C.HOME_XYZ is a Cartesian point
        # the arm is not near at the start, so driving there would be a
        # pointless detour through a pose the demonstration never occupies.
        E.move_to(env, C.HOME_XYZ, GRIP_OPEN, top_down=False, settle=0.3,
                  log=log, cfg=cfg, stats=stats)
    # ...and fold back to the exact starting configuration.  q_start is a
    # joint-space command, so there is no IK branch ambiguity to resolve: this
    # is the leg that makes the arm end where it began.
    E.sim(env, 2.0, arm=q_start, grip=GRIP_OPEN, hook=hook)
    E.sim(env, 0.4, arm=q_start, grip=GRIP_OPEN, hook=hook)
    back = float(np.max(np.abs(env.data.qpos[env._arm_qpos] - q_start)))
    stats["return_err_rad"] = round(back, 4)
    if back > 0.05:
        print(f"  !! return-to-start missed by {back:.3f} rad "
              f"({np.degrees(back):.1f} deg)", flush=True)
    stats["sim_time_s"] = env._sim_t - sim_
    return idx, ok, stats


_orig_move_to = E.move_to
_orig_solve = E._solve
_orig_sim = E.sim


def _round_vec(v, nd=6):
    """JSON-safe form for one entry of the episode vector."""
    if isinstance(v, np.ndarray):
        return [round(float(x), nd) for x in v.ravel()]
    if isinstance(v, (list, tuple)):
        return [round(float(x), nd) for x in v]
    return round(float(v), nd)


def apply_line_config(line):
    """Put the line back the way it was when an episode was collected.

    `line_config()` writes this into every manifest row precisely so that a
    replayer can reproduce the world.  The render and the evaluator build a
    RandEnv but never set RUNNING_LINE, so `_logic_and_belt` fell into its
    `elif` branch: whenever `conveyor_env._conveyor_logic` zeroed the belt -- a
    part arriving at the indexing stop, or being handled -- the surfacevel read
    0 and the elif deliberately did NOT re-assert, so the line stayed stopped.
    That is the OLD stopping line, and replays recorded on a running one missed
    the cube: the belt halted as the arm reached the part and the jaws closed
    on empty air, because the cube was no longer where the chase
    had aimed.

    Reads the episode's own record rather than a constant, so a dataset
    collected on a stopping line still renders as a stopping line.
    """
    if not line:
        return None
    globals()["RUNNING_LINE"] = bool(line.get("running", False))
    globals()["BELT_FIXED"] = line.get("belt_jitter") is None
    if line.get("belt_nominal"):
        C.BELT_SPEED = float(line["belt_nominal"])
    if line.get("spawn_every_s"):
        conveyor_env.SPAWN_EVERY = float(line["spawn_every_s"])
    _wl = line.get("wait_lead_s", None)
    if isinstance(_wl, (int, float)):
        globals()["WAIT_LEAD_S"] = float(_wl)
    elif isinstance(_wl, str) and line.get("wait_lead_default") is not None:
        # new way: lead varies per episode (vector wait_lead_s); the line
        # stores the range string + default. Replayers use the default as the
        # fallback and the episode's own vector value for the actual trigger.
        globals()["WAIT_LEAD_S"] = float(line["wait_lead_default"])
    if line.get("belt_narrow_var") is not None:
        globals()["BELT_NARROW_VAR"] = bool(line["belt_narrow_var"])
    if line.get("catch_iters") is not None:
        globals()["CATCH_ITERS"] = int(line["catch_iters"])
    if line.get("catch_band_mm") is not None:
        globals()["GRASP_CATCH_M"] = float(line["catch_band_mm"]) / 1000.0
    return dict(line)


def install_wait_pose(env):
    """Put the arm at the episode's START POSE, which is the wait pose.

    Every replayer has to call this or it replays a DIFFERENT episode.  The
    collector now begins each episode at the configured wait pose instead of HOME,
    and `RandEnv.reset` still writes HOME into qpos -- so a replayer that only
    resets starts the tape 1.9 rad away from where it was recorded (measured:
    `replay_dev 1.9e+00` where it had been 1.9e-07) and the rendered frames
    show an episode nobody collected.

    The pose travels in the episode's own `vector` (`wait_joints`), so the
    episode stays reproducible from what is stored, exactly like the stop and
    the tote offset.  Older datasets have no `wait_joints` and are unaffected:
    their episodes really do begin at HOME.
    """
    wj = env.vec.get("wait_joints") if env.vec else None
    if not wj:
        return False
    r = env.real
    # the clip bounds come from the MODEL, not from r._arm_lo/_arm_hi: those are
    # set by each caller, and a caller that forgets them gets
    # "'ConveyorPickEnv' object has no attribute '_arm_lo'".  Reading the model
    # keeps this helper independent of the caller.
    lo = r.model.actuator_ctrlrange[r._arm_aid, 0]
    hi = r.model.actuator_ctrlrange[r._arm_aid, 1]
    env._arm_lo, env._arm_hi = lo, hi      # __setattr__ forwards to the real env
    q = np.clip(np.asarray(wj, float), lo, hi)
    # qpos AND ctrl: qpos alone would be snapped back to a zero ctrl on the
    # first step of the replay
    r.data.qpos[r._arm_qpos] = q
    r.data.qpos[r._grip_qpos] = GRIP_OPEN
    r.data.ctrl[r._arm_aid] = q
    r.data.ctrl[r._grip_aid] = GRIP_OPEN
    mujoco.mj_forward(r.model, r.data)
    env.row0 = dict(
        t=0.0, qpos=r.data.qpos[r._arm_qpos].copy(),
        grip=r.data.qpos[r._grip_qpos],
        tcp=r._tcp(),
        cubes=np.stack([r._cube_pos(i) if r._active[i] else np.zeros(3)
                        for i in range(C.POOL)]),
        active=np.array(r._active, dtype=np.int8))
    return True


def full_close(env, grip):
    """A "close" command means close ALL the way, in the DATA.

    The demonstrations used to record GRIP_SHUT = 0.02 rad.  That is not a
    closed claw: it is merely where the servo stalls on a 32 mm part, so on an
    empty or missed grasp the jaws sat visibly ajar, and the rendered frames
    the inspector shows had a half-shut claw in them.  A real gripper is
    commanded shut and the OBJECT stops it, so the recorded command is the
    actuator's own lower bound (-0.17453 rad) and the +-2.94 N force limit
    does the stopping.

    It has to be here, in the command stream, and not in a viewer: patching it
    at display time made the viewer and the dataset show different claws, and patching it only in the render would make the pixels
    disagree with the recorded actions.  Measuring the outcome cannot see the
    difference -- with a part between the jaws the force limit stops the fingers
    in the same place either way -- but the DATA has to say what the robot was
    told to do, so this is where it belongs.
    """
    if grip is None or float(grip) > GRIP_SHUT + 1e-6:
        return grip
    return float(env.model.actuator_ctrlrange[env._grip_aid][0])


def _patched_sim(env, seconds, arm=None, grip=None, hook=None):
    """Route EVERY physics step through the collector's hook.

    `expert.move_to` calls `sim` internally without a hook; passing hook only
    at the episode level would log the belt-wait and grasp-settle steps and
    silently MISS the arm motion - most of the manipulation window.  The
    module-level hook makes the log exactly one row per physics step, which is
    also what makes the replay check meaningful.
    """
    grip = full_close(env, grip)
    return _orig_sim(env, seconds, arm=arm, grip=grip,
                     hook=hook if hook is not None else CTX.get("hook"))


def _patched_solve(ik, w, q, top_down, restarts, outer, iters):
    t0 = time.perf_counter()
    q, err, tilt = ik.solve(w, q0=q, top_down=top_down, yaw=CTX["jaw_yaw_vec"],
                            outer=outer, iters=iters, restarts=restarts)
    return q, err, tilt, time.perf_counter() - t0


AIM_RING_M = 0.0136            # jaw centring, measured
AIM_CLOS_M = 0.00185
# Grasp depth (Phase-7 hypothesis, FALSIFIED at n = 50 - kept for the record,
# the constant below is back to 0): the descent target is moved this far BELOW
# the part's centre.
# Phase-7 traces show the policy's closes landing 16-25 mm above the part
# (it fires the close during the descent), and the expert's measured vertical
# band is only about [-5, +10] mm around the part's centre.  Targeting 10 mm
# deep moves the whole demonstrated trajectory - and with it the policy's own
# early close - to dz = +6..+15 mm instead of +16..+25 mm.  At this depth the
# expert places 4/4 (10 mm z offset) with a clean 32 mm pinch, the
# blade's contact band straddles the part's centre of mass (a more stable
# carry), and the lowest nub still clears the belt by 3.5 mm.
# GRASP_DZ_M = 0.010   # FALSIFIED at n = 50 (PROVENANCE "deeper close")
GRASP_DZ_M = 0.0       # nominal tie, tier-B regression (14% vs 40% pick): back to 0


# ------------------------------------------------------- jaw-centring aim fix
def jaw_center_xy(jy_vec):
    """World-xy correction that aims the JAWS' centreline, not the site.

    The `grasp` site (build_cell) sits 13.6 mm off the jaw-tip midpoint along
    the ring axis, so aiming the site at the part's centre leaves the jaws
    that far off centre.  Measured before this fix:
    the blades only clipped the part's +ring corner and the expert's aim
    envelope was 0-3.5 mm - at +4 mm the close SHOVED the part 12 mm out of
    the jaws instead of pinching it (gap closed to 29.3 mm, part left on the
    belt).  With this correction the same episode places at EVERY aim offset
    from 0 to +20 mm (7/7), the close stalls at the part's own width
    (32.5-32.8 mm of a 32 mm part = a real pinch) and the residual shove
    falls from 9.7 to 3.4 mm as the error grows - the signature of a
    symmetric, self-centring push.

    The correction is a constant vector in the gripper frame, so it rotates
    with the commanded jaw yaw and is valid at every part yaw.
    """
    jy = np.asarray(jy_vec, float)[:2]
    n = float(np.linalg.norm(jy))
    if n < 1e-9:
        return np.zeros(2)
    jy = jy / n
    ring = np.array([-jy[1], jy[0]])
    return ring * AIM_RING_M + jy * AIM_CLOS_M


def _patched_move_to(env, target, grip, *a, **k):
    t = np.asarray(target, float).copy()
    # tote waypoints follow the per-episode box position.  THIS MUST COME
    # BEFORE the aim shift below: the aim/correction moves t[:2] by ~14 mm and
    # then the nominal tote centre no longer matches (measured: with the check
    # after the shift the drop target stayed at the OLD centre at every offset
    # and the arm descended into the moved box's rim).
    if (abs(float(t[0]) - demo_pick.TOTE_XY[0]) < 1e-9
            and abs(float(t[1]) - demo_pick.TOTE_XY[1]) < 1e-9):
        t[0] += CTX.get("tote_dx", 0.0)
        t[1] += CTX.get("tote_dy", 0.0)
    t[:2] += CTX["aim_xy"] + jaw_center_xy(CTX.get("jaw_yaw_vec", (0.0, 1.0)))
    if abs(float(t[2]) - APPROACH_Z) < 1e-9:
        t[2] += CTX["hover_dz"]
    elif abs(float(t[2]) - demo_pick.GRASP_Z) < 1e-9:
        t[2] -= GRASP_DZ_M          # 0 now; was the falsified deep grasp
    return _orig_move_to(env, t, grip, *a, **k)


E._solve = _patched_solve
E.move_to = _patched_move_to
E.sim = _patched_sim


# ------------------------------------------------------------ state logging
def make_hook(buf):
    def hook(env):
        r = env.real
        d = r.data
        ctrl = d.ctrl[r._arm_aid]
        grip = float(d.ctrl[r._grip_aid])
        buf["t"].append(r._sim_t)
        buf["qpos"].append(d.qpos[r._arm_qpos].copy())
        buf["grip"].append(d.qpos[r._grip_qpos])
        buf["tcp"].append(r._tcp())
        buf["cubes"].append(np.stack(
            [r._cube_pos(i) if r._active[i] else np.zeros(3)
             for i in range(C.POOL)]))
        buf["active"].append(np.array(r._active, dtype=np.int8))
        buf["act_raw"].append(np.append(ctrl, grip))
        buf["hash"].update(np.ascontiguousarray(d.qpos).tobytes())
        buf["hash"].update(np.ascontiguousarray(d.qvel).tobytes())
    return hook


def states_array(buf):
    """(T+1, ...) rows: row 0 is the post-reset state, row k the post-step-k."""
    q = np.asarray(buf["qpos"], dtype=np.float32)
    grip = np.asarray(buf["grip"], dtype=np.float32)[:, None]
    q0 = np.concatenate([np.asarray(buf["row0"]["qpos"], dtype=np.float32),
                         np.atleast_1d(np.float32(buf["row0"]["grip"]))])
    return dict(
        t=np.asarray([0.0] + buf["t"], dtype=np.float32),
        qpos=np.concatenate([q0[None], np.c_[q, grip]]).astype(np.float32),
        tcp=np.asarray([buf["row0"]["tcp"]] + buf["tcp"], dtype=np.float32),
        cubes=np.asarray([buf["row0"]["cubes"]] + buf["cubes"], dtype=np.float32),
        active=np.asarray([buf["row0"]["active"]] + buf["active"], dtype=np.int8),
    )


def normalized_actions(env, act_raw):
    m = env.real
    a = np.asarray(act_raw, dtype=np.float32).copy()
    a[:, :5] = (a[:, :5] - m._mid) / m._half
    glo, ghi = m._grip_lo, m._grip_hi
    a[:, 5] = (a[:, 5] - glo) / (ghi - glo) * 2.0 - 1.0
    return np.clip(a, -1.0, 1.0)


def replay_deviation(env, act_raw, states):
    """Re-drive the episode from the logged raw ctrl; max |dqpos| vs the log."""
    CTX.pop("hook", None)          # replay passes its own hook explicitly
    env.reset()
    # without this the replay starts at HOME while the episode started at the
    # wait pose, and the check reads 1.9 rad of "deviation" that is really just
    # the start pose (measured; it was 1.9e-07 before the start pose moved)
    install_wait_pose(env)
    dev = 0.0
    k = 1
    dt = env.real.model.opt.timestep

    def hook(e):
        nonlocal dev, k
        dev = max(dev, float(np.abs(
            e.real.data.qpos[e.real._arm_qpos] - states["qpos"][k][:5]).max()))
        k += 1

    for row in act_raw:
        E.sim(env, dt, arm=row[:5], grip=float(row[5]), hook=hook)
    return dev


# ------------------------------------------------------------------ workers
def _shared(fn, default=0):
    """Read a multiprocessing shared value, tolerating a dead kernel object.

    Windows creates these as named kernel objects, and a hard-killed process
    (taskkill /F /T, or a worker that dies holding the lock) leaves the name
    behind.  The next run then gets PermissionError [WinError 5] from
    `Event.is_set()`, the worker dies at the top of its loop, and the parent
    prints "0/160 ok  0 attempts" forever because it never checks exit codes.
    That is exactly what happened to ds_0008.  A shared counter that cannot be
    read is treated as "no progress yet"; it can only cause a redundant
    episode, never a wrong one, because the seed comes from `ep_id`.
    """
    try:
        return int(fn() if callable(fn) else fn)
    except (OSError, ValueError):
        return default


def worker_main(wid, args, counter, successes, lock, stop, codeh):
    np.set_printoptions(suppress=True)
    # The wait-pose override has to be re-applied HERE.  main() sets the module
    # global, but the pool is mp.get_context("spawn"): on Windows each worker is
    # a fresh interpreter that re-imports this module, so the global comes back
    # as None and every worker silently collected the DEFAULT pose.  Measured
    # that way ds_0007 came out byte-identical to ds_0006 (act_raw equal for
    # ep 0), which is how it was caught.  args is pickled into the child, so
    # the value does cross the boundary.
    if getattr(args, "wait_joints", None):
        globals()["WAIT_JOINTS"] = list(args.wait_joints)
        if wid == 0:
            print(f"[w0] wait pose override active: "
                  f"{[round(x, 3) for x in args.wait_joints]}", flush=True)
    # same spawn trap as WAIT_JOINTS above: the worker pool is mp spawn, so the
    # global is None in every child unless it is re-applied here from args
    if getattr(args, "running_line", False):
        globals()["RUNNING_LINE"] = True
        if getattr(args, "belt_speed", None):
            C.BELT_SPEED = float(args.belt_speed)
        if getattr(args, "spawn_every", None):
            conveyor_env.SPAWN_EVERY = float(args.spawn_every)
        if wid == 0:
            print(f"[w0] running line: belt {C.BELT_SPEED:.3f} m/s, "
                  f"spawn every {conveyor_env.SPAWN_EVERY:.1f} s", flush=True)
    if getattr(args, "belt_fixed", False):
        globals()["BELT_FIXED"] = True
        globals()["BELT_NARROW_VAR"] = False
    if getattr(args, "belt_narrow_var", None) is not None:
        globals()["BELT_NARROW_VAR"] = bool(args.belt_narrow_var)
        if wid == 0:
            print(f"[w0] belt narrow var {globals()['BELT_NARROW_VAR']} "
                  f"U[{BELT_VAR_LO_MPS},{BELT_VAR_HI_MPS}]", flush=True)
    if getattr(args, "no_wait_jitter", False):
        globals()["_WAIT_JITTER_OFF"] = True
    env = RandEnv(ConveyorPickEnv())
    out = args.out
    os.makedirs(out, exist_ok=True)
    man_path = os.path.join(out, "manifest.jsonl")

    done_ids = set()
    if os.path.exists(man_path):
        with open(man_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["ep_id"])
                except Exception:
                    pass

    def next_ep():
        for _ in range(3):
            try:
                with lock:
                    i = counter.value
                    counter.value += 1
                return i
            except (OSError, ValueError):
                time.sleep(0.5)
        return _shared(counter.value)

    def _stopped(ev):
        try:
            return ev.is_set()
        except (OSError, ValueError):
            return False

    while not _stopped(stop) and _shared(successes.value) < args.target:
        if args.attempt_cap and _shared(counter.value) > args.attempt_cap:
            break
        ep = next_ep()
        if ep in done_ids:
            continue
        seed = BASE_SEED + ep
        env.vec = sample_vector(seed)
        # THE START POSE TRAVELS WITH THE EPISODE (jittered per-episode).
        # Every replayer installs it after reset or it replays a different
        # episode (measured 1.9 rad deviation). Jitter is slight by design:
        # a bigger move would put the pick out of reach.
        if getattr(args, "wait_joints", None):
            base = list(args.wait_joints)
            if globals().get("_WAIT_JITTER_OFF"):
                env.vec["wait_joints"] = [float(x) for x in base]
            else:
                vrng = np.random.default_rng(
                    (int(seed) ^ 0x51F7 ^ SEED_SALT) & 0xFFFFFFFF)
                env.vec["wait_joints"] = sample_wait_joints(base, vrng)
        CTX.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in env.vec.items()})
        buf = dict(t=[], qpos=[], grip=[], tcp=[], cubes=[], active=[],
                   act_raw=[], row0=None, hash=hashlib.sha256())
        hook = make_hook(buf)
        CTX["hook"] = hook
        t0 = time.perf_counter()
        idx, ok, stats = episode_stable_wait(env, seed=seed, cfg=E.FAST, log=False,
                                   hook=hook, stats=dict(ik_s=0.0, solves=0,
                                                         sim_s=0.0, steps=0,
                                                         reseeds=0))
        wall = time.perf_counter() - t0
        CTX.pop("hook", None)
        buf["row0"] = env.row0
        reason = ("no part" if idx is None else
                  "grasp failed" if not ok and not env.real._in_tote(
                      env.real._cube_pos(idx)) and env.real._grasped() != idx
                  else "not in tote" if not ok else "ok")

        row = dict(ep_id=ep, seed=seed, worker=wid, ok=bool(ok),
                   picked=int(idx) if idx is not None else -1,
                   reason=reason,
                   look_off_px=stats.get("look_off_px"),
                   # ndarray -> list, list -> list of rounded floats (the start
                   # pose `wait_joints` is a list and plain round() raised
                   # "type list doesn't define __round__", which killed every
                   # worker the moment the start pose moved into the vector),
                   # everything else -> a single rounded float
                   vector={k: (_round_vec(v))
                           for k, v in env.vec.items()},
                   n_steps=len(buf["t"]), wall_s=round(wall, 2),
                   ik_s=round(stats["ik_s"], 2), solves=stats["solves"],
                   sim_s=round(stats.get("sim_time_s", 0.0), 2),
                   code=codeh, npz=None, line=line_config())

        if ok:
            states = states_array(buf)
            act_raw = np.asarray(buf["act_raw"], dtype=np.float64)
            act = normalized_actions(env, act_raw)
            dev = replay_deviation(env, act_raw, states)
            row["replay_dev_m"] = float(f"{dev:.3e}")
            row["replay_ok"] = bool(dev < 1e-4)
            name = f"ep_{ep:06d}.npz"
            np.savez_compressed(
                os.path.join(out, name),
                **states, act=act, act_raw=act_raw.astype(np.float32),
                vector=json.dumps(row["vector"]), ep=np.int64(ep),
                seed=np.int64(seed))
            row["npz"] = name
        with lock:
            successes.value += 1 if ok else 0
            with open(man_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"[w{wid}] ep {ep:4d}  {'OK ' if ok else 'DROP'} {reason:14s} "
              f"wall {wall:5.2f}s  steps {len(buf['t']):4d}"
              + (f"  replay_dev {row.get('replay_dev_m', float('nan')):.1e}"
                 if ok else ""), flush=True)
    env.real.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target", type=int, default=300,
                    help="successful episodes to keep")
    ap.add_argument("--out", default=current.DS)
    ap.add_argument("--attempt-cap", type=int, default=0,
                    help="hard cap on attempts (0 = 3x target)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wait-joints", type=float, nargs=5,
                    default=list(current.WAIT_JOINTS),
                    metavar=("PAN", "LIFT", "ELBOW", "WFLEX", "WROLL"),
                    help="park the arm at these 5 joint angles during the wait; "
                         "the DEFAULT is the current wait pose (the episode "
                         "begins and ends there). Pass --wait-home to collect "
                         "the OLD way, which starts at HOME instead.")
    ap.add_argument("--wait-home", action="store_true",
                    help="OLD WAY: start the episode at HOME instead of the "
                         "wait pose")
    ap.add_argument("--running-line", action="store_true", default=True,
                    help="default True: the belt NEVER stops. The indexing "
                         "stop is overridden every physics step and the pick "
                         "leads a moving part. See RUNNING_LINE.")
    ap.add_argument("--stopping-line", action="store_true",
                    help="OLD WAY: let the belt stop at the indexing station")
    ap.add_argument("--belt-speed", type=float, default=current.BELT_MPS,
                    help="C.BELT_SPEED in m/s; default is the measured 0.08. "
                         "Without --belt-jitter the episode vector does NOT "
                         "multiply it, so every episode runs at exactly this.")
    ap.add_argument("--belt-jitter", action="store_true",
                    help="OLD WAY: multiply the belt speed by U[0.6, 1.4] per "
                         "episode. At the current limit that puts 0.13 m/s -- "
                         "past where the arm starts dropping boxes -- in some "
                         "episodes.")
    ap.add_argument("--belt-fixed", action="store_true", default=False,
                    help="OLD WAY: no per-episode belt-speed variance at all "
                         "(exact 0.08 every episode). Default is now the narrow "
                         "variance U[0.068,0.092]; pass --belt-fixed to "
                         "reproduce ds_0010 exactly.")
    ap.add_argument("--belt-narrow-var", dest="belt_narrow_var",
                    action="store_true", default=True,
                    help="NEW WAY default: per-episode belt U[0.068,0.092] "
                         "around 0.08 nominal (+-15%%, inside the arm budget).")
    ap.add_argument("--no-belt-narrow-var", dest="belt_narrow_var",
                    action="store_false",
                    help="disable narrow belt variance (with --belt-fixed for "
                         "exact ds_0010 reproduction)")
    ap.add_argument("--no-wait-jitter", action="store_true",
                    help="disable per-episode wait-pose jitter "
                         "(reproduce ds_0010 exact start pose)")
    ap.add_argument("--spawn-every", type=float, default=current.SPAWN_EVERY_S,
                    help="seconds between parts entering the line; default is "
                         "the current 8.0 s")
    args = ap.parse_args()
    # THE NEW WAY IS THE DEFAULT.  Every one of these used to be an opt-in flag,
    # which is the same trap as the bench ignoring --ds: one forgotten flag and
    # you are collecting a dataset under the old contract without noticing.  The
    # old way is still reachable, but only by naming it (--stopping-line,
    # --belt-jitter, --wait-home).
    if args.stopping_line:
        args.running_line = False
        print("OLD WAY: --stopping-line, the belt will stop at the station")
    if args.belt_jitter:
        args.belt_fixed = False
        print("OLD WAY: --belt-jitter, per-episode speed U[0.6, 1.4]")
    if args.wait_home:
        args.wait_joints = None
        print("OLD WAY: --wait-home, episodes start at HOME")
    if args.belt_narrow_var and args.belt_jitter:
        print("NOTE: --belt-jitter (OLD 0.6-1.4x) overrides --belt-narrow-var")
        args.belt_narrow_var = False
    globals()["BELT_NARROW_VAR"] = bool(args.belt_narrow_var)
    if args.running_line:
        globals()["RUNNING_LINE"] = True
        if args.belt_speed:
            C.BELT_SPEED = float(args.belt_speed)
        if args.spawn_every:
            conveyor_env.SPAWN_EVERY = float(args.spawn_every)
        else:
            conveyor_env.SPAWN_EVERY = float(current.SPAWN_EVERY_S)
        _belt_tag = ("fixed" if args.belt_fixed else
                     f"narrow U[{BELT_VAR_LO_MPS},{BELT_VAR_HI_MPS}]"
                     if args.belt_narrow_var else "x U[0.6, 1.4]")
        print(f"running line: belt {C.BELT_SPEED:.3f} m/s ({_belt_tag}), "
              f"spawn every {conveyor_env.SPAWN_EVERY:.1f} s")
    if args.belt_fixed:
        globals()["BELT_FIXED"] = True
        globals()["BELT_NARROW_VAR"] = False
        C.BELT_SPEED = float(args.belt_speed)
    if getattr(args, "no_wait_jitter", False):
        globals()["_WAIT_JITTER_OFF"] = True
        print("OLD WAY: --no-wait-jitter, exact wait pose every episode")
    if getattr(args, "wait_joints", None):
        import collect_state as _self
        _self.WAIT_JOINTS = list(args.wait_joints)
        print(f"wait pose override (joint space): "
              f"{[round(x, 3) for x in args.wait_joints]}")
    if args.smoke:
        args.workers, args.target = 1, 3
        args.out = "ds_smoke"
    args.attempt_cap = args.attempt_cap or 3 * args.target

    print(f"mujoco {mujoco.__version__}  torch "
          f"{__import__('torch').__version__}")
    codeh = code_hash()
    print(f"code hash {codeh}  out {args.out}  workers {args.workers}  "
          f"target {args.target}")

    os.makedirs(args.out, exist_ok=True)
    man_path = os.path.join(args.out, "manifest.jsonl")
    done = 0
    if os.path.exists(man_path):
        with open(man_path) as f:
            done = sum(1 for line in f if json.loads(line).get("ok"))
    if done >= args.target:
        print(f"already have {done} ok episodes in {args.out}; nothing to do")
        return 0

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", sum(1 for _ in open(man_path)) if
                       os.path.exists(man_path) else 0)
    successes = ctx.Value("i", done)
    lock = ctx.Lock()
    stop = ctx.Event()
    t0 = time.perf_counter()
    procs = [ctx.Process(target=worker_main,
                         args=(w, args, counter, successes, lock, stop, codeh))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    while any(p.is_alive() for p in procs):
        time.sleep(5.0)
        el = time.perf_counter() - t0
        ok, att = _shared(successes.value), _shared(counter.value)
        # exit code 0 = a worker that reached the target and left cleanly;
        # only a NON-zero code is a crash worth shouting about
        dead = [p.exitcode for p in procs
                if not p.is_alive() and p.exitcode not in (0, None)]
        print(f"  ... {ok}/{args.target} ok  "
              f"{att} attempts  {el/60:.1f} min  "
              f"({ok/max(el, 1e-9)*60:.0f}/h)"
              + (f"  [dead workers: {dead}]" if dead else ""), flush=True)
        # a dead worker means nothing else will ever increment the counters;
        # report it instead of printing 0/N for the rest of the budget
        if all(not p.is_alive() for p in procs):
            print("  every worker has exited; stopping", flush=True)
            break
    for p in procs:
        if not p.is_alive() and p.exitcode not in (0, None):
            print(f"  worker exited with code {p.exitcode}", flush=True)
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    print(f"\ndone: {successes.value} ok / {counter.value} attempts  "
          f"wall {wall/60:.2f} min  (budget 15 min)")
    print(f"projection for 3000 episodes: {wall/successes.value*3000/60:.0f} min"
          if successes.value else "no episodes")
    return 0 if successes.value >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
