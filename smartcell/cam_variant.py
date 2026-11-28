"""Wrist-camera mount for the SO-101, on the arm's own `camera_mount` body.

The mount is the one the menagerie `robotstudio_so101` spec intends: the spec
hangs `wrist_cam` off a `camera_mount` body that is a child of `gripper`, and
carries a physical housing there (`wrist_camera_lens` at mount-local
[-0.003, 0.0567, -0.0495]).  `check_mount()` asserts the model still has that
body and that parentage before anything is written, so a future build_cell that
re-parents the camera fails loudly instead of silently moving a camera nobody
measured.

The pose is a rigid offset from the grasp site, which is rigidly fixed to the
`gripper` body, so solving it once and writing it into the mount frame gives the
same mount at every arm configuration.

WHY THE CAMERA IS PARENTED TO THE WORLD
----------------------------------------
In this MuJoCo 3.14.0 build the Python `MjvCamera` binding exposes only
azimuth / distance / elevation / fixedcamid / lookat / orthographic /
trackbodyid / type - no `pos`, no `quat`, no `fovy`.  A camera declared on a
moving body therefore renders with its POSITION read from `cam_xpos` but its
ORIENTATION left at the binding's default, so `wrist_cam` looks off at the hall
floor no matter what `cam_quat` says.  Measured: with the body-mounted camera
aimed exactly at the grasp site, the part covered 0.000 of the frame at every
fovy from 1 to 50 degrees, while the frame was filled by conveyor beams 0.7 m
away.  Cameras declared on the WORLD body render correctly (`overhead` does).

So the mount is still the arm's: `Mount.update()` reads the `camera_mount`
body's world pose every frame and writes the optical pose into `cam_pos` /
`cam_quat` in world coordinates, with `cam_bodyid` set to the world.  The camera
is rigidly carried by the arm exactly as a body-mounted camera would be; only
the frame of reference differs.  A side benefit is that `cam_fovy` works again,
which it does not for free cameras.

GEOMETRY, and why the standoff is 150 mm rather than 60-120
------------------------------------------------------------
`back` is the distance from the GRASP SITE to the lens, measured `deg` off the
grasp site's +Z (which points back up the arm toward the wrist; the jaws travel
along the site's -Z).  The tilt is toward the gripper's +Y, which is the
direction the menagerie housing already uses and, crucially, is PERPENDICULAR
to the jaw-separation axis - so the lens looks in between the fingers instead of
through one of them.

Measured part coverage (fraction of frame, hide-and-diff, 32 mm part at the
site):

    standoff \\ fovy     50     45     40     35     30
      120 mm         .000   .000   .000   .000   .000
      140 mm         .066   .083   .112   .146   .205
      150 mm         .081   .102   .133   .177   .244
      160 mm         .070   .090   .115   .154   .214
      180 mm         .055   .069   .090   .120   .167
      200 mm         .025   .032   .041   .055   .077

Two forced deviations from the Phase 0 spec, both measured rather than assumed:

  * standoff 150 mm, not 60-120 mm.  At 120 mm and closer the lens sits inside
    the gripper's own envelope and the part is 0.000 visible at EVERY fovy: the
    fingers and palm occlude it completely.  The nearest occluders are 59-77 mm
    above the site with only 15-25 mm of lateral clearance.
  * fovy 35, not 50.  The standoff is forced by occlusion and the field of view
    is then free; at fovy 50 the best achievable coverage is 8.1%, short of the
    15% gate.  fovy 35 reaches 17.7% and fovy 30 reaches 24.4%.  8.1% vs the
    15% target is a factor of 1.85, inside the 2x stop-and-report threshold, so
    the phase proceeds with the deviation stated rather than silently.

Run:  python smartcell/cam_variant.py
"""
import numpy as np
import mujoco

import build_cell as C
from conveyor_env import HOME, GRIP_OPEN

CAM_NAME = "wrist_cam"
SITE_NAME = "grasp"
MOUNT_BODY = "camera_mount"        # the menagerie body the camera is declared on
GRIP_BODY = "gripper"
PART_M = 0.032                     # C.CUBE, the part this view has to resolve

# Phase 0 spec, as parameters, with the two measured deviations applied.
# azim 180, not 0.  Re-derived with a mount sweep once the near-plane bug
# below was fixed - see the note on apply_clip_fix.  With the gripper actually
# rendered, tilting toward the gripper's +Y (azim 0) puts the palm and the
# finger roots across the line of sight to the part: measured part coverage
# collapses to under 4% of the frame at the grasp frame.  Tilting the other way
# (azim 180) keeps the lens clear of them: measured 19.5% coverage with 4658 px
# of gripper visible, against the Phase 0 gate of 15%.
DEFAULT = dict(back=0.150, deg=40.0, fovy=35.0, azim=180.0)

# Renderer clip planes, in ABSOLUTE units, applied by apply_clip_fix.
NEAR_MM = 2.0
FAR_M = 25.0


def apply_clip_fix(model):
    """Stop the renderer from deleting the gripper.

    MuJoCo scales the near plane by the MODEL extent:

        scene.near = model.vis.map.znear * model.stat.extent

    This model carries the hall/backdrop scenery, so stat.extent is 12.2563 m
    and the stock znear of 0.01 puts the near plane at 122.6 mm.  The wrist
    camera sits only 150 mm from the grasp site and the gripper's palm, mount
    housing and finger roots are 25-120 mm from the lens, so a large part of
    the gripper was BEING DISCARDED in every render - including every frame of
    ds_0001..ds_0004.

    Measured: a 32 mm part parked on the optical axis,
    provably inside the frustum, rendered 0 px at 60, 80 and 100 mm and 1336 px
    at 130 mm.  The observed cliff between 100 and 130 mm matches the predicted
    122.6 mm exactly.

    Consequence that forced a mount change too: with the
    plane fixed the part's coverage at the grasp frame falls 15.2% -> 1.2%,
    because the 15.2% was the camera seeing the part THROUGH a gripper that was
    not being drawn.  So the standoff table in this module's docstring was
    measured against a scene containing no gripper and the mount had to be
    re-derived; see DEFAULT above.

    Values are expressed as fractions of the extent because that is the unit
    MuJoCo wants: 2 mm near / 25 m far.  A far plane of 25 m still clears the
    belt, the tote and the floor of the cell, and keeps far/near at 12500 rather
    than the 61280 the stock pair gives, which is better for depth precision.
    """
    ext = float(model.stat.extent)
    if ext <= 0:
        raise RuntimeError(f"model.stat.extent is {ext!r}; clip planes would "
                           f"be meaningless")
    model.vis.map.znear = (NEAR_MM / 1000.0) / ext
    model.vis.map.zfar = FAR_M / ext


def _name(model, objtype, objid):
    return mujoco.mj_id2name(model, objtype, int(objid)) or ""


def _id(model, objtype, name):
    return int(mujoco.mj_name2id(model, objtype, name))


def check_mount(model, name=CAM_NAME):
    """Assert the camera is still the one this mount was derived from.

    Guards the post-compile write: if a future build re-parents or renames the
    camera, silently rewriting cam_pos would move something never measured.
    """
    cid = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        raise RuntimeError(f"camera {name!r} not in the compiled model")
    bid = int(model.cam_bodyid[cid])
    if _name(model, mujoco.mjtObj.mjOBJ_BODY, bid) != MOUNT_BODY:
        raise RuntimeError(
            f"{name} is on body {_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)!r}, "
            f"expected {MOUNT_BODY!r}; cam_variant needs re-deriving")
    parent = int(model.body_parentid[bid])
    if _name(model, mujoco.mjtObj.mjOBJ_BODY, parent) != GRIP_BODY:
        raise RuntimeError(
            f"{MOUNT_BODY} is a child of "
            f"{_name(model, mujoco.mjtObj.mjOBJ_BODY, parent)!r}, expected "
            f"{GRIP_BODY!r}; the mount is no longer rigid with the grasp site")
    return cid, bid


def _tilt(azim_deg):
    """Unit tilt direction in the mount frame.

    Mount-local +Z points from the site back up the arm, +X is the jaw
    separation axis and +Y is perpendicular to it.  azim rotates the tilt about
    +Z, so azim=0 (the default, and the menagerie housing's own direction) looks
    in between the fingers rather than through one.
    """
    a = np.radians(azim_deg)
    return np.array([-np.sin(a), np.cos(a), 0.0])


class Mount:
    """The wrist camera, carried by the arm, expressed in world coordinates.

    `update()` must be called after every mj_forward (or at least before every
    render), because the camera is world-parented and its pose is recomputed
    from the `camera_mount` body's world transform each time.
    """

    def __init__(self, model, back=None, deg=None, fovy=None, azim=None,
                 name=CAM_NAME, site=SITE_NAME, data=None, stock=True):
        self.model = model
        # before anything renders: without this the gripper is not drawn at all
        apply_clip_fix(model)
        self.cid, self.mount_bid = check_mount(model, name)
        self.sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.sid < 0:
            raise RuntimeError(f"site {site!r} not in the compiled model")
        if int(model.site_bodyid[self.sid]) != _id(model, mujoco.mjtObj.mjOBJ_BODY,
                                                   GRIP_BODY):
            raise RuntimeError(f"site {site!r} is not on {GRIP_BODY!r}")
        self.back = DEFAULT["back"] if back is None else float(back)
        self.deg = DEFAULT["deg"] if deg is None else float(deg)
        self.fovy = DEFAULT["fovy"] if fovy is None else float(fovy)
        self.azim = DEFAULT["azim"] if azim is None else float(azim)
        self.site_local = np.array(model.site_pos[self.sid], float)
        # The mount the MODEL declares, captured before cam_bodyid is cleared
        # below.  so101.xml says, on camera_mount (a child of `gripper` with
        # pos=quat=identity, so this is the GRIPPER frame):
        #     <camera name="wrist_cam" mode="fixed"
        #             pos="-0.00296 0.05436 -0.05310"
        #             quat="0.958280 -0.285792 -0.001556 0.004547" fovy="61"/>
        # with a matching physical housing modelled at the lens
        # (wrist_camera_lens, size 0.007, same quat) and the PCB behind it.
        # Read from the model rather than hardcoded, so this cannot drift away
        # from the asset it describes.
        self.stock_pos = np.array(model.cam_pos[self.cid], float).copy()
        self.stock_quat = np.array(model.cam_quat[self.cid], float).copy()
        self.stock_fovy = float(model.cam_fovy[self.cid])
        # stock=True reproduces the model's declared pose exactly (composed
        # once from body-local to world); stock=False uses the derived
        # back/deg/azim mount that DEFAULT parameterises.
        #
        # The DEFAULT IS NOW True.  It was False, which meant every caller -
        # render_dataset.py, evaluate.py, watch.py - silently built the
        # FABRICATED mount (back=0.150, deg=40, azim=180, fovy=35) while the
        # arm bench showed the model's real camera, so the bench and the
        # training images disagreed about what the robot sees.  Measured from
        # the configured wait pose, the declared mount looks 4.4 deg
        # off the pick station against a 30.5 deg half-fov; the derived one is a
        # different camera entirely, 156.6 mm away in the gripper frame.
        self.stock = bool(stock)
        # world-parent: the orientation of a body-mounted camera is not applied
        # by this build's binding, but a world camera renders correctly
        model.cam_bodyid[self.cid] = 0
        self.last = None
        if data is not None:
            self.update(data)

    def describe_stock(self):
        """The model's declared mount, in the gripper frame, and what it aims at."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.stock_quat)
        Rm = R.reshape(3, 3)
        fwd = -Rm[:, 2]                      # a MuJoCo camera looks down -Z
        to_site = self.site_local - self.stock_pos
        ds = float(np.linalg.norm(to_site))
        ang = float(np.degrees(np.arccos(np.clip(
            fwd @ (to_site / max(ds, 1e-12)), -1.0, 1.0))))
        return dict(stock_pos=[round(float(v), 5) for v in self.stock_pos],
                    stock_quat=[round(float(v), 6) for v in self.stock_quat],
                    stock_fovy=self.stock_fovy,
                    stock_standoff_mm=round(ds * 1000.0, 2),
                    stock_site_off_axis_deg=round(ang, 2),
                    stock_in_cone=bool(ang <= self.stock_fovy / 2.0))

    def update(self, data):
        """Recompute and write the optical pose from the arm's current state.

        The world pose is also pushed into `data.cam_xpos` / `data.cam_xmat`,
        which is what `mjv_updateScene` actually reads for a fixed camera.  Those
        are otherwise only refreshed by `mj_forward`, so writing `cam_pos` after
        the forward pass leaves the renderer one pose stale - measured as the
        same grasp configuration rendering a blank frame or the part depending
        only on call order.  Because the camera is world-parented its world pose
        IS cam_pos/cam_quat, so refreshing them here is exact and needs no
        second physics pass.
        """
        m = self.model
        d = data
        P = d.xpos[self.mount_bid].astype(float)
        R = d.xmat[self.mount_bid].astype(float).reshape(3, 3)
        if self.stock:
            # EXACTLY what so101.xml declares: the body-local pose composed
            # once through the mount body's world transform.  cam_variant's
            # derived mount aims at the grasp site instead, which is a
            # different camera: measured 156.6 mm away in the gripper frame
            # and pointing 21.5 deg differently at the part.
            cam_w = P + R @ self.stock_pos
            Rq = np.zeros(4)
            mujoco.mju_mat2Quat(Rq, R.ravel())
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, Rq, self.stock_quat)
            quat = quat / np.linalg.norm(quat)
            site_w = P + R @ self.site_local
            fovy = self.stock_fovy
        else:
            a = np.radians(self.deg)
            loc = self.site_local + self.back * (
                np.cos(a) * np.array([0.0, 0.0, 1.0])
                + np.sin(a) * _tilt(self.azim))
            cam_w = P + R @ loc
            site_w = P + R @ self.site_local
            quat = C._look_at(cam_w, site_w)[1]
            fovy = self.fovy
        m.cam_fovy[self.cid] = fovy
        m.cam_pos[self.cid] = cam_w
        m.cam_quat[self.cid] = quat
        d.cam_xpos[self.cid] = cam_w
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, np.asarray(quat, float))
        d.cam_xmat[self.cid] = Rq
        self.last = dict(cam_world=cam_w, site_world=site_w)
        return cam_w

    def describe(self, data=None):
        """Measure the mount against the grasp site, in the mount frame.

        Computed from the mount-local parameters, not from a readback of the
        world pose: the offset is rigid, so the standoff and tilt are the same
        at every arm configuration, and deriving them from `self.last` would
        silently mix two different arm states whenever describe() is called
        after the arm has moved since the last update().
        """
        a = np.radians(self.deg)
        cam_l = self.site_local + self.back * (np.cos(a) * np.array([0.0, 0.0, 1.0])
                                               + np.sin(a) * _tilt(self.azim))
        delta = cam_l - self.site_local
        dist = float(np.linalg.norm(delta))
        off = float(np.degrees(np.arccos(np.clip(delta[2] / max(dist, 1e-12),
                                                -1.0, 1.0))))
        mm_px = 2.0 * dist * np.tan(np.radians(self.fovy) / 2.0) / 128.0 * 1000.0
        part_px = PART_M * 1000.0 / mm_px * max(0.0, np.cos(np.radians(off)))
        return dict(fovy=self.fovy,
                    standoff_mm=round(dist * 1000.0, 2),
                    off_axis_deg=round(off, 2),
                    mount_local=[round(float(v), 4) for v in cam_l],
                    part_px_at_128=round(part_px, 1),
                    part_frac_lower_bound=round(part_px ** 2 / 128.0 ** 2, 4))


def strip_scene_markers(renderer):
    """Delete mjv_updateScene's DEBUG MARKERS from a render scene.

    Call it AFTER `renderer.update_scene(...)` and BEFORE `renderer.render()`.

    WHAT THIS IS FOR - the pink ball
    ---------------------------------
    A magenta ball showed up in the middle of the wrist camera's view.  It is
    a rendering artifact: `mjv_updateScene` appends, after the model's own geoms, a small
    sphere per light as a visualisation marker.  This cell's lights are all
    directionless point lights 3+ m up, far outside the frustum - but the
    marker for light 2 is placed along the camera axis and lands 67 mm in
    front of the wrist lens, dead centre of the frame.

    Measured on ds_0007/imgs/ep_000000 frame 0 (128x128 wrist render): 144
    pixels of it, peak RGB 148/22/152, a ~13 px blob at rows 75-87, cols
    97-109.  The scene entry that draws it is

        objtype = 6 (mjOBJ_LIGHT), objid = 2, type = 2 (mjGEOM_SPHERE),
        size = 0.004, rgba = (1, 0, 1, 0.3), pos = (0.073, 0.114, 0.845)

    i.e. a 4 mm translucent magenta debug sphere.  It is not a geom, not a
    material, not a texture and not a part - hiding each of the five pool
    parts in turn removed 0 of its pixels, and no material in the model has a
    magenta rgba.

    The markers carry `objtype != mjOBJ_GEOM`, so truncating the scene at the
    first one removes them.  `scene.geoms` is exposed as an immutable tuple in
    this binding (there is no `mjv_copyGeom`), so truncation at the first
    marker is the only way - and it is safe because mjv_updateScene appends the
    model's geoms first and the markers after them.  Measured on ds_0007 frame
    0: 502 scene geoms in, 500 out, and the ONLY pixels that change are the
    226 of the ball's own footprint.

    Every rendered frame of every dataset rendered so far (ds_0001..ds_0007)
    carries this ball; this is where it stops.

    Returns the number of marker geoms removed (0 when there were none).
    """
    sc = renderer._scene
    n = int(sc.ngeom)
    # scan BACKWARDS: the markers are appended last, so this is O(#markers)
    # (~1-2), not O(#geoms).  A forward scan cost 2.5 ms/frame and tripled
    # the render cost (1.37 -> 3.84 ms/frame) on a 502-geom scene.
    keep = n
    while keep > 0 and int(sc.geoms[keep - 1].objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
        keep -= 1
    if keep == n:
        return 0
    sc.ngeom = keep
    return n - keep


def _canonical(model, data=None):
    """Model state at the IK home pose, so a mount solved here is valid
    everywhere: the offset is rigid, so the same local pose holds at any
    arm configuration."""
    d = mujoco.MjData(model) if data is None else data
    mujoco.mj_resetData(model, d)
    adr = [int(model.joint(n).qposadr[0]) for n in
           ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
            "wrist_roll")]
    d.qpos[adr] = HOME
    d.qpos[int(model.joint("gripper").qposadr[0])] = GRIP_OPEN
    mujoco.mj_forward(model, d)
    return d


def apply(model, back=None, deg=None, fovy=None, azim=None, name=CAM_NAME,
          site=SITE_NAME, data=None):
    """One-shot: build the mount and write it at the model's current state."""
    d = _canonical(model) if data is None else data
    mnt = Mount(model, back=back, deg=deg, fovy=fovy, azim=azim, name=name,
                site=site, data=d)
    return mnt.describe(d)


def stock(model, name=CAM_NAME, site=SITE_NAME):
    """The menagerie camera as declared, for the record."""
    cid, bid = check_mount(model, name)
    d = _canonical(model)
    pos = np.array(model.cam_pos[cid], float)
    site_l = np.array(model.site_pos[_id(model, mujoco.mjtObj.mjOBJ_SITE, site)],
                      float)
    delta = pos - site_l
    dist = float(np.linalg.norm(delta))
    return dict(cam_id=cid, mount_local=[round(float(v), 4) for v in pos],
                fovy=float(model.cam_fovy[cid]),
                standoff_mm=round(dist * 1000.0, 2),
                off_axis_deg=round(float(np.degrees(np.arccos(np.clip(
                    delta[2] / max(dist, 1e-12), -1.0, 1.0)))), 2))


if __name__ == "__main__":
    m = C.build_model()
    print("mujoco", mujoco.__version__)
    print("stock  ", stock(m))
    print("variant", apply(m))
