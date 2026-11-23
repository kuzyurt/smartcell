"""Damped-least-squares IK for the SO-101 grasp site.

Used to prove the cell is solvable (reach checks) and to drive the scripted
pick-and-place demo.  The 5 arm hinges are solved; the gripper slide is left
to the caller.
"""
import mujoco
import numpy as np

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# The grasp site carries an identity quat, so its +Z is the gripper body's +Z,
# which points *against* the approach direction.  A top-down grasp therefore
# means site +Z == world +Z, with the jaws travelling down along -Z.
SITE_Z_UP = np.array([0.0, 0.0, 1.0])


class ArmIK:
    """Solver state is kept in a private MjData: solve() writes qpos while it
    iterates, and sharing the caller's data would teleport the arm mid-sim."""

    def __init__(self, model, data=None, site="grasp"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = int(model.site(site).id)
        self.adr = np.array([int(model.joint(n).qposadr[0]) for n in ARM])
        self.jid = [int(model.joint(n).id) for n in ARM]
        self.lo = model.jnt_range[self.jid, 0].copy()
        self.hi = model.jnt_range[self.jid, 1].copy()
        self.grip = int(model.joint("gripper").qposadr[0])
        self.grip_hi = float(model.jnt_range[int(model.joint("gripper").id)][1])

    def pose(self, q):
        self.d.qpos[self.adr] = np.clip(q, self.lo, self.hi)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return (self.d.site_xpos[self.site].copy(),
                self.d.site_xmat[self.site].reshape(3, 3).copy())

    def _dls(self, J, b, lam):
        return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(J.shape[0]), b)

    def _wrist_task(self, q, yaw):
        """3-row angular task on the two wrist joints: point down, and square
        the jaw axis to `yaw`.  The jaw axis is the gripper body's +X, so this
        is what decides whether the jaws close on the part's faces or on its
        corners."""
        p, r = self.pose(q)
        tilt = np.cross(r[:, 2], SITE_Z_UP)[:2]
        yaw_err = float(np.cross(r[:, 0], np.asarray(yaw, float))[2])
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
        J = jr[:, self.adr[3:]]
        b = np.array([tilt[0], tilt[1], yaw_err])
        return J, b, float(np.hypot(tilt[0], tilt[1])), abs(yaw_err)

    def solve(self, target, q0=None, top_down=True, yaw=(0.0, 1.0, 0.0),
              outer=30, iters=40, tol=1e-4, lam=0.008, restarts=8, rng=None):
        """Joint angles that put the grasp site on `target` (world metres).

        The 3 proximal joints place the wrist, the 2 wrist joints aim the
        jaws, and the two stages are alternated because aiming moves the tip.
        With top_down the site's +Z is driven to world +Z, so the jaws travel
        straight down, and `yaw` is the world direction the jaw axis (site +X)
        is aligned to.  Candidate solutions are ranked by position error first
        and then by distance from q0, so a move continues from the branch the
        arm is already in instead of flipping the elbow mid-trajectory.
        Returns (q, position_error_m, tilt_deg); tilt is NaN if not top_down.
        """
        target = np.asarray(target, float)
        rng = np.random.default_rng(0) if rng is None else rng
        near = None if q0 is None else np.asarray(q0, float)
        seeds = []
        if q0 is not None:
            seeds.append(np.asarray(q0, float))
        seeds += [rng.uniform(0.3 * self.lo, 0.3 * self.hi) for _ in range(restarts)]
        best = None
        for q0s in seeds:
            q = np.clip(np.asarray(q0s, float), self.lo, self.hi)
            for _ in range(outer if top_down else 1):
                if top_down:
                    for _ in range(iters):
                        J, b, tilt, ye = self._wrist_task(q, yaw)
                        if max(tilt, ye) < 1e-3:
                            break
                        q[3:] = np.clip(q[3:] + self._dls(J, b, lam),
                                        self.lo[3:], self.hi[3:])
                for _ in range(iters):
                    p, r = self.pose(q)
                    e = target - p
                    if np.linalg.norm(e) < tol:
                        break
                    jp = np.zeros((3, self.m.nv))
                    jr = np.zeros((3, self.m.nv))
                    mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
                    q[:] = np.clip(q + self._dls(jp[:, self.adr], e, lam),
                                   self.lo, self.hi)
            p, r = self.pose(q)
            err = float(np.linalg.norm(target - p))
            ang = float(np.degrees(np.arccos(np.clip(r[:, 2] @ SITE_Z_UP, -1, 1))))
            yaw_err = abs(float(np.cross(r[:, 0], np.asarray(yaw, float))[2]))
            dist = 0.0 if near is None else float(np.abs(q - near).max())
            key = (round(err, 5) + 0.05 * round(yaw_err, 4),
                   ang if top_down else 0.0, dist)
            if best is None or key < best[0]:
                best = (key, q.copy(), err, ang)
        _, q, err, ang = best
        return q, err, (ang if top_down else float("nan"))

    def margin(self, q):
        """Smallest distance to a joint limit, radians."""
        q = np.clip(np.asarray(q, float), self.lo, self.hi)
        return float(np.minimum(q - self.lo, self.hi - q).min())

    def jaw_gap(self, q):
        """Distance between the two jaw tip spheres for a given arm pose."""
        self.d.qpos[self.adr] = np.clip(q, self.lo, self.hi)
        mujoco.mj_kinematics(self.m, self.d)
        a = self.d.geom_xpos[int(self.m.geom("fixed_jaw_sph_tip1").id)]
        b = self.d.geom_xpos[int(self.m.geom("moving_jaw_sph_tip1").id)]
        return float(np.linalg.norm(a - b))

    def grasp_qpos(self, opening=0.019):
        """Gripper slide value that centres the jaws on a part of width 2*opening."""
        lo, hi = 0.0, self.grip_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            self.d.qpos[self.grip] = mid
            self.d.qpos[self.adr] = np.zeros(5)
            mujoco.mj_kinematics(self.m, self.d)
            a = self.d.geom_xpos[int(self.m.geom("fixed_jaw_sph_tip1").id)]
            b = self.d.geom_xpos[int(self.m.geom("moving_jaw_sph_tip1").id)]
            if np.linalg.norm(a - b) < 2 * opening:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def set_qpos(self, q, grip=None):
        """Write an arm pose (and optional gripper slide) into the shared data."""
        self.d.qpos[self.adr] = np.clip(np.asarray(q, float), self.lo, self.hi)
        if grip is not None:
            self.d.qpos[self.grip] = float(grip)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy()
