"""Opt-in motion smoothing for the frozen smartcell controller.

The validated protocol (``evaluate.py`` defaults) is untouched. This module
changes *how the arm command reaches the actuators*, not what the policy
decides:

* ``ramp``   -- the policy emits one arm target per 40 ms entry and
  ``expert.sim`` holds it constant for all 8 physics steps, so the position
  servo sees a staircase that steps at 25 Hz. With ``ramp`` the target is
  instead interpolated linearly from the previous target to the new one over
  the entry.
* ``park_*`` / ``ema`` -- existing ``evaluate.py`` knobs (latch the command
  while parked at the wait pose, first-order smoothing).

Use ``install(name)`` before calling ``evaluate.rollout``; ``install("off")``
restores the validated behaviour.
"""
import numpy as np
import mujoco

import evaluate as EV
import expert as E

PROFILES = {
    "off": dict(ramp=0.0, park_r=0.0, park_db=0.02, ema=0.0),
    # interpolate each 40 ms target: no extra lag, about -57 % joint jerk
    "smooth": dict(ramp=1.0),
    # additionally filter the command and latch the parked tremor: calmer, ~20 % slower
    "silky": dict(ramp=1.0, park_r=0.1, ema=0.5),
}

_orig_sim = E.sim
_state = dict(ramp=0.0)
RECORD = None      # set to a list to capture arm joint angles at every physics step


def _ramped_sim(env, seconds, arm=None, grip=None, hook=None):
    ramp = _state["ramp"]
    if arm is None or (ramp <= 0.0 and RECORD is None):
        return _orig_sim(env, seconds, arm=arm, grip=grip, hook=hook)
    m, d = env.model, env.data
    dt = m.opt.timestep
    n = int(seconds / dt)
    start = d.ctrl[env._arm_aid].copy()
    goal = np.clip(arm, env._arm_lo, env._arm_hi)
    k_ramp = max(1, int(round(n * ramp)))
    for k in range(n):
        a = min(1.0, (k + 1) / k_ramp) if ramp > 0.0 else 1.0
        d.ctrl[env._arm_aid] = start + (goal - start) * a
        if grip is not None:
            d.ctrl[env._grip_aid] = grip
        mujoco.mj_step(m, d)
        env._sim_t += dt
        env._conveyor_logic()
        if RECORD is not None:
            RECORD.append(d.qpos[env._arm_qpos].copy())
        if hook is not None:
            hook(env)


def install(profile):
    """Apply a profile (name or dict) to the evaluate/expert modules."""
    p = dict(PROFILES["off"])
    p.update(PROFILES[profile] if isinstance(profile, str) else profile)
    EV.ARM_PARK_R = float(p["park_r"])
    EV.ARM_PARK_DB = float(p["park_db"])
    EV.ARM_EMA = float(p["ema"])
    _state["ramp"] = float(p["ramp"])
    E.sim = _ramped_sim
    return p
