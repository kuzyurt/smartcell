"""Assembly-line pick-and-place environment for vision RL.

The conveyor is the official MuJoCo `surfacevel` belt (zero DOF, no actuators):
it runs because its surface moves, so this file only steps physics and manages
the parts.  Loop: parts appear on the infeed pallet, are placed on the belt at
the infeed, ride +X through the arm's verified grasp window, and drop off the
head roller into the discharge bin if missed.  The arm picks a part and places
it in the tote on the table.

Obs: overhead camera image + arm proprio + part positions.
"""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import build_cell as C
from arm_ik import ArmIK

IMG = 128
POLICY_DT = 0.05
# IK home: gripper high on the +X side, clear of the grasp window and the tote
HOME = np.radians([-26.0, -43.4, 44.8, 8.6, 23.4])
GRIP_OPEN = 0.64                 # 66 mm jaw gap, i.e. open.  A narrower gap does
                                  # not work: fixed_jaw_box3, the plate that carries
                                  # the fixed finger, sits 7.5 mm below the grasp
                                  # site at the tip's y, so at 55 mm the plate lands
                                  # on the part's top corner and the arm stalls
                                  # 30 mm above it.  Opened to 66 mm the plate runs
                                  # 8 mm outside the faces and the fingers can pass
                                  # the corner.  The site is the tip midpoint at the
                                  # 0.50 slide, so the jaws close symmetrically.
GRIP_SHUT = 0.02                 # 17 mm closed: the servo stalls on the part
SPAWN_EVERY = 3.0
# Parts allowed in play at once.  This is a budget, not a limit on the scene:
# pick_cycle caps the tote at 3 and keeps 2 more in flight so the belt always
# has one part arriving while the arm is carrying the last one (a part takes
# ~8 s to reach the pick station, a pick-and-place takes ~3.5 s).  It has to be
# at least cap + 2 or the tote filling up would starve the spawner and the
# line would stop feeding itself.  C.POOL is 5, so nothing is ever parked for
# want of a slot.
MAX_ACTIVE = 5
MAX_STEPS = 600
TOTE_C = np.array(C.TOTE_XY)
# Servo-indexing line: the drive holds a part at the pick station instead of
# letting it run past, which is how a pick-and-place conveyor actually works and
# is the only way a 160 mm grasp window can serve a moving part.  If the arm
# does not take it within HOLD_T, the line indexes on and the part is lost.
X_STOP = 0.10                    # the drive holds the part once its centre passes this
HOLD_T = 8.0                    # how long the line waits for the robot


def set_belt_speed(model, speed):
    """Write the surface velocity of the official belt and its two rollers."""
    for name in ("belt", "roller_tail", "roller_head"):
        g = int(model.geom(name).id)
        sv = model.geom_surfacevel[g]
        if name == "belt":
            sv[0] = speed
        else:
            sv[5] = -speed / C.ROLLER_R


class ConveyorPickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", None]}

    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.model = C.build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model

        arm = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll")
        self._arm_joints = list(arm)
        self._arm_qpos = [int(m.joint(n).qposadr[0]) for n in arm]
        self._arm_aid = [int(m.actuator(n).id) for n in arm]
        self._grip_aid = int(m.actuator("gripper").id)
        self._grip_qpos = int(m.joint("gripper").qposadr[0])
        lo = m.actuator_ctrlrange[self._arm_aid, 0]
        hi = m.actuator_ctrlrange[self._arm_aid, 1]
        self._mid = 0.5 * (lo + hi)
        self._half = 0.5 * (hi - lo)
        self._grip_lo, self._grip_hi = m.actuator_ctrlrange[self._grip_aid]

        self._cube_qpos = [int(m.jnt_qposadr[int(m.body(f"cube_{i}").jntadr[0])])
                           for i in range(C.POOL)]
        self._cube_dof = [int(m.jnt_dofadr[int(m.body(f"cube_{i}").jntadr[0])])
                          for i in range(C.POOL)]
        self._jaw_f = int(m.geom("fixed_jaw_sph_tip1").id)
        self._jaw_m = int(m.geom("moving_jaw_sph_tip1").id)
        self._grasp_site = int(m.site("grasp").id)
        self._cube_bodies = [{int(m.body(f"cube_{i}").id)} for i in range(C.POOL)]
        # take the finger bodies from the tip geoms: the fixed finger is rigidly
        # part of the palm body, so there is no "fixed_jaw" body to look up.
        self._jaws_ids = [int(m.geom_bodyid[g]) for g in (self._jaw_f, self._jaw_m)]
        self._overhead = int(m.camera("overhead").id)
        self._nsub = max(1, int(round(POLICY_DT / m.opt.timestep)))
        self.ik = ArmIK(m)

        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (IMG, IMG, 3), np.uint8),
            "qpos": spaces.Box(-np.inf, np.inf, (6,), np.float64),
            "tcp": spaces.Box(-np.inf, np.inf, (3,), np.float64),
            "cubes": spaces.Box(-np.inf, np.inf, (C.POOL, 3), np.float64),
            "active": spaces.Box(0, 1, (C.POOL,), np.int8),
        })
        self._renderer = mujoco.Renderer(self.model, height=IMG, width=IMG)
        # shadows off for the policy camera: they cost 4.6x per obs render
        # (measured 220 ms vs 48 ms) and the top-down policy image does not
        # need them.  The interactive viewer and every presentation camera
        # (verify_cell, demo_pick, cell_view) keep their shadows.
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._sim_t = 0.0
        self._last_spawn = -1e9
        self._active = [False] * C.POOL
        self._placed = set()
        self._success = False
        self._steps = 0
        self._hold_t = 0.0
        self._held_at_stop = -1
        set_belt_speed(self.model, C.BELT_SPEED)

    # ------------------------------------------------------------------ api
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[self._arm_qpos] = HOME
        d.qpos[self._grip_qpos] = GRIP_OPEN
        for i in range(C.POOL):
            self._park(i)
        self._active = [False] * C.POOL
        self._placed = set()
        self._success = False
        self._steps = 0
        self._sim_t = 0.0
        self._last_spawn = -1e9
        self._spawn()
        self._hold_t = 0.0
        self._held_at_stop = -1
        set_belt_speed(self.model, C.BELT_SPEED)
        mujoco.mj_forward(self.model, d)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(6), -1.0, 1.0)
        arm = self._mid + self._half * action[:5]
        grip = self._grip_lo + (action[5] + 1.0) * 0.5 * (self._grip_hi - self._grip_lo)
        for _ in range(self._nsub):
            self.data.ctrl[self._arm_aid] = arm
            self.data.ctrl[self._grip_aid] = grip
            mujoco.mj_step(self.model, self.data)
            self._sim_t += self.model.opt.timestep
            self._conveyor_logic()
        self._steps += 1
        obs = self._obs()
        rew, success = self._reward(obs)
        return obs, rew, False, self._steps >= MAX_STEPS, {"success": success}

    # -------------------------------------------------------------- mechanics
    def _park(self, i):
        d = self.data
        a = self._cube_qpos[i]
        d.qpos[a:a + 3] = [C.PARK[0] + i * C.PARK_DX, C.PARK[1],
                           C.PARK[2] + C.CUBE / 2]
        d.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[self._cube_dof[i]:self._cube_dof[i] + 6] = 0.0
        self._active[i] = False

    def _spawn(self):
        for i in range(C.POOL):
            if not self._active[i]:
                d = self.data
                a = self._cube_qpos[i]
                d.qpos[a:a + 3] = [C.SPAWN_X, C.BELT_Y,
                                   C.BELT_TOP + C.CUBE / 2 + 0.01]
                d.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
                d.qvel[self._cube_dof[i]:self._cube_dof[i] + 6] = 0.0
                self._active[i] = True
                self._last_spawn = self._sim_t
                return True
        return False

    def _conveyor_logic(self):
        """Part lifecycle plus the indexing stop at the pick station."""
        d = self.data
        if (self._sim_t - self._last_spawn >= SPAWN_EVERY
                and sum(self._active) < min(MAX_ACTIVE, C.POOL)):
            self._spawn()
        for i in range(C.POOL):
            if not self._active[i]:
                continue
            a = self._cube_qpos[i]
            if d.qpos[a] > C.BELT_END_X or d.qpos[a + 2] < C.DESPAWN_Z:
                self._park(i)

        held = self._grasped()
        at_stop = None
        for i in range(C.POOL):
            if not self._active[i] or i == held:
                continue
            if (d.qpos[self._cube_qpos[i]] > X_STOP
                    and abs(d.qpos[self._cube_qpos[i] + 1] - C.BELT_Y) < 0.06):
                at_stop = i
                break
        if at_stop is None:
            self._hold_t = 0.0
            self._held_at_stop = -1
            set_belt_speed(self.model, C.BELT_SPEED)
            return
        if at_stop != self._held_at_stop:
            self._held_at_stop = at_stop
            self._hold_t = 0.0
        self._hold_t += self.model.opt.timestep
        if self._hold_t > HOLD_T:            # not taken: index on, part lost
            self._hold_t = 0.0
            self._held_at_stop = -1
            set_belt_speed(self.model, C.BELT_SPEED)
        else:
            set_belt_speed(self.model, 0.0)

    def _tcp(self):
        return self.data.site_xpos[self._grasp_site].copy()

    def _jaws(self):
        d = self.data
        return d.geom_xpos[self._jaw_f].copy(), d.geom_xpos[self._jaw_m].copy()

    def _cube_pos(self, i):
        return self.data.qpos[self._cube_qpos[i]:self._cube_qpos[i] + 3].copy()

    def _jaw_bodies(self):
        """Body ids of the two fingers (the fixed finger is part of the palm)."""
        return self._jaws_ids

    def _grasped(self):
        """Index of a part held between the jaws, else None.

        The gripper is force limited, so when it clamps a part the slide stalls
        at the part width instead of reaching its command.  The test is the
        real thing: the jaws are within a few mm of shut and the part is in
        contact with both fingers, i.e. friction on both faces and no yaw.
        """
        f, m = self._jaws()
        if float(np.linalg.norm(f - m)) > 0.040:
            return None
        d = self.data
        jaws = self._jaw_bodies()
        touching = [set(), set()]
        for c in d.contact:
            b1 = int(c.geom1)
            b2 = int(c.geom2)
            for k, jb in enumerate(jaws):
                bg = int(self.model.geom_bodyid[b1]) == jb
                bg2 = int(self.model.geom_bodyid[b2]) == jb
                if not (bg or bg2):
                    continue
                other = b2 if bg else b1
                ob = int(self.model.geom_bodyid[other])
                for i in range(C.POOL):
                    if not self._active[i]:
                        continue
                    if ob in self._cube_bodies[i]:
                        touching[k].add(i)
        for i in touching[0] & touching[1]:
            return i
        return None

    def _in_tote(self, p):
        x0, x1, y0, y1 = C.TOTE_INNER
        return (x0 < p[0] < x1 and y0 < p[1] < y1
                and C.TOTE_FLOOR < p[2] < C.TOTE_RIM)

    # ------------------------------------------------------------------ obs
    def _obs(self):
        d = self.data
        self._renderer.update_scene(d, camera=self._overhead)
        img = self._renderer.render()
        cubes = np.zeros((C.POOL, 3))
        for i in range(C.POOL):
            cubes[i] = self._cube_pos(i) if self._active[i] else 0.0
        return {
            "image": img.copy(),
            "qpos": np.array([d.qpos[a] for a in self._arm_qpos]
                             + [d.qpos[self._grip_qpos]]),
            "tcp": self._tcp(),
            "cubes": cubes,
            "active": np.array(self._active, dtype=np.int8),
        }

    def _reward(self, obs):
        tcp = obs["tcp"]
        act = [i for i in range(C.POOL) if self._active[i]]
        dmin = min((float(np.linalg.norm(obs["cubes"][i] - tcp)) for i in act),
                   default=0.5)
        r = -0.5 * min(dmin, 0.5) - 0.01
        held = self._grasped()
        if held is not None:
            r += 1.0
            r += 0.5 * (1.0 - min(float(np.linalg.norm(
                obs["cubes"][held][:2] - TOTE_C)) / 0.3, 1.0))
        for i in act:
            if i not in self._placed and self._in_tote(obs["cubes"][i]):
                self._placed.add(i)
                self._success = True
                r += 10.0
        return r, bool(self._success)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._obs()["image"]
        return None

    def close(self):
        self._renderer.close()




