"""Fine mechanical detail for the assembly line: drive unit, fasteners, decals.

build_cell.py lays out the machine and line_frame.py builds its structure.
This module puts the hardware on it - the gearbox and motor down to their fins,
the coupling, the terminal box, the sight glass and the nameplate, the socket
screws that hold the extrusion together and the side guides on, the retaining
caps on the rollers, and the control cabinet with its e-stop, lamp, hinges and
decals.  A conveyor built only from frame members reads as a diagram, and the
whole point of the cell is that it should read as a machine somebody could
service.

ANCHORING: everything here hangs off the far side beam's OUTER face, which is
conveyor_drive.FRAME_OUT (0.370) - the frame got 50 mm thicker when the flat
8 mm side plates became real extrusions, so the whole drive train moved 42 mm
outboard with it.  The x coordinates (GX0/GX1/MX) are unchanged: only the mount
plane moved.

FOUR RULES, all of them learned from verify_cell.py failing:

1. EVERY geom created here is VISUAL ONLY - contype = conaffinity = 0.  It
   cannot collide, so it cannot perturb the physics that verify_cell.py and
   demo_pick.py gate on.  The moving belt, the picking pose and the part pool
   are untouched by anything in this file.

2. NOTHING IS BURIED INSIDE ANOTHER PART.  verify_cell.py section 2 fails on
   any static pair closer than -0.5 mm, so a bolt cannot have a shank that
   disappears into the plate it passes through, and a motor fin cannot start
   inside the motor barrel.  Detail butts against the surfaces it is bolted
   to; it never occupies the same space.  That is why bolt() draws a head and
   a socket recess and no shank.

3. NOTHING FLOATS.  Section 3 requires every static geom to touch the floor or
   another part.  The base plates and levelling feet that used to live here are
   now line_frame.feet(), which builds them as one supported stack.

4. ROTATIONS MUST BE REAL QUATERNIONS.  A rotation about x is
   (cos t/2, sin t/2, 0, 0) - the vector part goes in slots 1..3, never slot 0,
   which is the scalar.  Getting that wrong silently produces zero quaternions,
   and a zero-quaternion geom lands somewhere absurd and then reads as an
   impossible interference.

Layout is passed in as a dict by build_cell.py rather than imported, so there
is exactly one source of truth for the coordinates.
"""
import mujoco
import numpy as np

from mjbox import box as _box, col

VIS = dict(contype=0, conaffinity=0)     # rule 1, used by every helper here

BOLT_R = 0.0045        # M8 head radius
BOLT_HEAD = 0.0035     # head height
SOCKET_R = 0.0022      # hex socket recess

# the drive train, all derived from the extrusion it bolts to
FRAME_OUT = 0.370              # far side beam's outer face (build_cell.FRAME_Y)
GY0, GY1 = FRAME_OUT, FRAME_OUT + 0.052        # gearbox, outboard of the beam
GZ0, GZ1 = 0.545, 0.660
GX0, GX1 = 0.700, 0.800
GYC = 0.5 * (GY0 + GY1)        # 0.396, the drive centreline
MR = 0.040                     # motor barrel radius
MX = 0.936                     # motor centre, x
M0, M1 = MX - 0.070, MX + 0.070

CAST = (0.40, 0.43, 0.45, 1.0)
BRIGHT = (0.86, 0.87, 0.89, 1.0)


# --------------------------------------------------------------- orientations
_AXIS_Q = {                      # cylinder local axis is +z, rotate onto x / y
    "x": np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]),
    "y": np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]),
    "z": np.array([1.0, 0.0, 0.0, 0.0]),
}
_AXIS_V = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
           "z": np.array([0, 0, 1.0])}
_AXIS_I = {"x": 1, "y": 2, "z": 3}     # rule 4: vector slots are 1..3


def _rot(axis, ang):
    """Unit quaternion for `ang` radians about x, y or z."""
    h = ang / 2.0
    v = np.zeros(4)
    v[0] = np.cos(h)
    v[_AXIS_I[axis]] = np.sin(h)
    return v


def _cyl(parent, name, pos, axis, r, half, rgba, quat=None, sgn=1.0):
    """A cylinder whose flat face at `pos` is the mounting face.

    `pos` is a point ON the surface the part is bolted to and the cylinder
    grows away from it along `axis` * sgn.  `sgn=-1` is for fasteners on a
    south-facing panel.  Rule 2 lives here: nothing is ever placed so that
    it ends up inside the part it is attached to.
    """
    d = _AXIS_V[axis] * sgn
    g = parent.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = col([r, half, 0.0])
    g.pos = col(np.asarray(pos, float) + d * half)
    g.quat = col(quat if quat is not None else _AXIS_Q[axis])
    g.rgba = col(list(rgba))
    g.contype = 0
    g.conaffinity = 0
    return g


# ------------------------------------------------------------------ fasteners
def bolt(parent, name, pos, axis="z", rgba=BRIGHT, r=BOLT_R, head=BOLT_HEAD,
         sgn=1.0):
    """A hex-socket cap screw: a head proud of the face, and its socket.

    Two geoms, not three.  A drawn shank would have to pass through the
    part it holds, which rule 2 forbids; the dark socket is what actually
    reads as "a screw" at this scale anyway.
    """
    d = _AXIS_V[axis] * sgn
    p = np.asarray(pos, float)
    _cyl(parent, f"{name}_h", p, axis, r, head / 2, rgba, sgn=sgn)
    _cyl(parent, f"{name}_k", p + d * head, axis, SOCKET_R, 0.0009,
         (0.13, 0.13, 0.14, 1.0), sgn=sgn)
    return name


# ------------------------------------------------------------------ the build
def build(spec, W, L):
    """Dress the line.  `L` is build_cell's layout dict, passed in verbatim."""
    n0 = len(spec.geoms)
    drv(spec, W, L)          # gearbox, coupling, motor, its pedestal
    screws(spec, W, L)       # socket screws along the extrusions
    rollers(spec, W, L)      # end caps and the bearing-block screws
    cabinet(spec, W, L)      # e-stop, hinges, latch, decals
    return len(spec.geoms) - n0


# ------------------------------------------------------------- drive assembly
def drv(spec, W, L):
    """Gearbox, flexible coupling, motor, pedestal - the whole drive train.

    Laid out along one axis with nothing overlapping: every part starts
    exactly where the previous one ends (rule 2).  The motor sits 136 mm
    outboard of the gearbox face so a 66 mm coupling fits between them.
    """
    CONV_Z = L["CONV_Z"]
    bed0, bed1 = CONV_Z - MR - 0.030, CONV_Z - MR - 0.014      # 0.530, 0.546

    # ---- gearbox: cooling ribs on the upper outboard face.  They stop
    # short of the sight glass and the nameplate below them, which is how a
    # real casting is laid out.
    for k, xr in enumerate(((0.712, 0.720), (0.732, 0.740), (0.752, 0.760),
                            (0.772, 0.780))):
        _box(W, f"drv_rib_{k}", xr, (GY1, GY1 + 0.007), (0.622, 0.646),
             rgba=CAST, **VIS)

    # breather plug on the top face, clear of the terminal box at x 0.72-0.78
    _cyl(W, "drv_breather", (0.790, GYC, GZ1), "z", 0.006, 0.007,
         (0.70, 0.71, 0.73, 1.0))
    _cyl(W, "drv_breather_cap", (0.790, GYC, GZ1 + 0.014), "z", 0.008, 0.002,
         (0.55, 0.57, 0.60, 1.0))

    # oil sight glass in a machined boss, below the ribs
    _cyl(W, "drv_sight_boss", (0.730, GY1, 0.585), "y", 0.014, 0.004,
         (0.50, 0.52, 0.55, 1.0))
    _cyl(W, "drv_sight_glass", (0.730, GY1 + 0.008, 0.585), "y", 0.010, 0.002,
         (0.72, 0.78, 0.66, 1.0))

    # maker's nameplate, riveted to the casting
    _box(W, "nameplate_gb", (0.752, 0.778), (GY1, GY1 + 0.0012),
         (0.602, 0.618), rgba=(0.88, 0.89, 0.90, 1.0), **VIS)
    for i, (bx, bz) in enumerate(((0.755, 0.607), (0.775, 0.607),
                                  (0.755, 0.613), (0.775, 0.613))):
        bolt(W, f"screw_np_{i}", (bx, GY1 + 0.0012, bz), "y", r=0.0018,
             head=0.0012)

    # ---- base flange and its anchors: the flange lands ON the far beam's
    # outer face (y = 0.370), which is the rib face of the extrusion's T-slot
    # above z 0.523 - reaching inboard of it would drive the flange 50 mm into
    # the beam, which now has depth in y rather than being a 8 mm sheet
    _box(W, "drv_flange", (GX0 - 0.012, GX1 + 0.012), (GY0, GY1 + 0.008),
         (GZ0 - 0.008, GZ0), rgba=(0.50, 0.52, 0.55, 1.0), **VIS)
    for i, (bx, by) in enumerate(((GX0 + 0.006, GY0 + 0.004),
                                  (GX1 - 0.006, GY0 + 0.004),
                                  (GX0 + 0.006, GY1 - 0.004),
                                  (GX1 - 0.006, GY1 - 0.004))):
        bolt(W, f"bolt_flange_{i}", (bx, by, GZ0 - 0.008), "z", sgn=-1)

    # ---- output shaft, then the coupling: four parts end to end
    _cyl(W, "drv_shaft", (GX1, GYC, CONV_Z), "x", 0.011, 0.011, BRIGHT)
    _cyl(W, "drv_hub_a", (0.822, GYC, CONV_Z), "x", 0.026, 0.009, CAST)
    _cyl(W, "drv_spider", (0.840, GYC, CONV_Z), "x", 0.022, 0.004,
         (0.78, 0.44, 0.10, 1.0))          # orange elastomer
    _cyl(W, "drv_hub_b", (0.848, GYC, CONV_Z), "x", 0.026, 0.009, CAST)
    for i, hx in enumerate((0.831, 0.857)):
        bolt(W, f"screw_setsh_{i}", (hx, GYC, CONV_Z + 0.026), "z", r=0.003,
             head=0.0025)

    # ---- motor barrel is cv_motor in build_cell; its fins are the single
    # biggest cue that this is a motor and not a plain cylinder.  They are
    # tangent to the barrel at r = MR, not sunk into it (rule 2).
    for k in range(16):
        th = 2 * np.pi * k / 16
        rad = np.array([0.0, -np.sin(th), np.cos(th)])
        f = W.add_geom()
        f.name = f"mot_fin_{k}"
        f.type = mujoco.mjtGeom.mjGEOM_BOX
        f.size = col([0.036, 0.0016, 0.005])       # axial, tangential, radial
        f.pos = col(np.array([MX, GYC, CONV_Z]) + rad * 0.045)
        f.quat = col(_rot("x", th))
        f.rgba = col(list(CAST))
        f.contype = 0
        f.conaffinity = 0

    # drive-end shield, cowl with a grille, encoder - again end to end.
    # COWL_FACE is the cowl's outboard end; the grille and the encoder both
    # start there, so nothing floats off it.
    _cyl(W, "mot_shield", (M1, GYC, CONV_Z), "x", MR + 0.004, 0.008, CAST)
    _cyl(W, "fincap_cowl", (M1 + 0.016, GYC, CONV_Z), "x", MR + 0.004, 0.022,
         CAST)
    COWL_FACE = M1 + 0.060          # _cyl grows outward from the face it is
                                   # given, so the cowl's outboard end is
                                   # pos + 2*half = M1+0.016 + 0.044
    for k in range(8):                     # grille bars on the cowl's face
        th = 2 * np.pi * k / 8
        rad = np.array([0.0, -np.sin(th), np.cos(th)])
        s = W.add_geom()
        s.name = f"fincap_slot_{k}"
        s.type = mujoco.mjtGeom.mjGEOM_BOX
        s.size = col([0.0016, 0.003, 0.012])
        s.pos = col(np.array([COWL_FACE + 0.0016, GYC, CONV_Z]) + rad * 0.026)
        s.quat = col(_rot("x", th))
        s.rgba = col([0.16, 0.17, 0.18, 1.0])
        s.contype = 0
        s.conaffinity = 0
    _cyl(W, "enc_body", (COWL_FACE, GYC, CONV_Z), "x", 0.010, 0.011,
         (0.28, 0.30, 0.33, 1.0))
    _cyl(W, "enc_cap", (COWL_FACE + 0.022, GYC, CONV_Z), "x", 0.008,
         0.0015, (0.34, 0.36, 0.39, 1.0))

    # ---- feet, bed plate and pedestal.  The feet stop exactly on the
    # barrel at z = CONV_Z - MR; the pedestal stops exactly on the bed.
    for k, (fx0, fx1) in enumerate(((M0 + 0.010, M0 + 0.030),
                                    (M1 - 0.030, M1 - 0.010))):
        _box(W, f"mot_foot_{k}", (fx0, fx1), (GY0 - 0.006, GY1 + 0.002),
             (bed1, CONV_Z - MR), rgba=CAST, **VIS)
        for j, by in enumerate((GY0 + 0.004, GY1 - 0.004)):
            bolt(W, f"bolt_motfoot_{k}{j}",
                 (0.5 * (fx0 + fx1), by, CONV_Z - MR), "z")
    _box(W, "mot_bed", (M0 + 0.006, M1 - 0.006), (GY0 - 0.010, GY1 + 0.006),
         (bed0, bed1), rgba=(0.46, 0.48, 0.51, 1.0), **VIS)
    # the drive overhangs the end of the frame (the beams stop at x = 0.800,
    # the motor runs to 1.006), so the bed needs its own pedestal.  It is
    # 42 mm clear of the discharge bin's north wall (y <= 0.325) and 76 mm
    # clear of the frame's end in x, so the two never touch.
    _box(W, "mot_pedestal", (M0 + 0.010, M1 - 0.010),
         (GY0 + 0.007, GY1 + 0.005), (0.008, bed0),
         rgba=(0.46, 0.48, 0.51, 1.0), **VIS)
    _box(W, "mot_pedestal_plate", (M0, M1), (GY0 - 0.003, GY1 + 0.015),
         (0.0, 0.008), rgba=(0.58, 0.60, 0.63, 1.0), **VIS)
    # The anchor bolts straddle the pedestal: the plate is 70 mm wide, the
    # pedestal 50 mm, so a 10 mm bolt radius fits exactly in each 10 mm
    # margin -- its head lands flush on the plate's edge and clear of the
    # pedestal's flank.
    for k, (bx, by) in enumerate(((M0 + 0.012, GY0 + 0.002),
                                  (M1 - 0.012, GY0 + 0.002),
                                  (M0 + 0.012, GY1 + 0.010),
                                  (M1 - 0.012, GY1 + 0.010))):
        bolt(W, f"bolt_ped_{k}", (bx, by, 0.008), "z", r=0.005, head=0.004)

    # ---- terminal box: a lid and four screws, and a blind cable gland on
    # its west face (a blank entry, so nothing has to hang off it)
    tx0, tx1 = 0.720, 0.780
    ty0, ty1, tz1 = GYC - 0.014, GYC + 0.014, 0.700
    _box(W, "mot_tlid", (tx0 + 0.004, tx1 - 0.004), (ty0 + 0.004, ty1 - 0.004),
         (tz1, tz1 + 0.003), rgba=(0.44, 0.46, 0.49, 1.0), **VIS)
    for i, (bx, by) in enumerate(((tx0 + 0.008, ty0 + 0.008),
                                  (tx1 - 0.008, ty0 + 0.008),
                                  (tx0 + 0.008, ty1 - 0.008),
                                  (tx1 - 0.008, ty1 - 0.008))):
        bolt(W, f"screw_tlid_{i}", (bx, by, tz1 + 0.003), "z", r=0.003,
             head=0.0022)
    _cyl(W, "cable_gland", (tx0, GYC, 0.680), "x", 0.009, 0.010,
         (0.50, 0.52, 0.55, 1.0), sgn=-1)
    _cyl(W, "cable_gland_cap", (tx0 - 0.020, GYC, 0.680), "x", 0.011, 0.005,
         (0.56, 0.58, 0.61, 1.0), sgn=-1)


# --------------------------------------------- socket screws on the extrusion
def screws(spec, W, L):
    """The two fastener rows that make the frame read as an assembly.

    One cap screw every 160 mm down each beam's outer face, sunk into the
    extrusion's upper rib (z 0.523..0.575, so the head sits on solid metal
    rather than bridging the T-slot).  This is the row a person looking at a
    real conveyor sees, and it is the single cheapest thing that stops a
    50 x 145 profile from reading as a painted box.

    The x run stops clear of the drive unit's flange (x 0.688..0.812 on the far
    side) and of the end caps.
    """
    pitch = 0.160
    x0, x1 = L["FRAME_X0"] + 0.040, L["FRAME_X1"] - 0.040
    z = 0.548
    n = int((x1 - x0) / pitch)
    for tag, (ya, yb), s in (("s", L["FRAME_Y"][0], -1.0),
                             ("n", L["FRAME_Y"][1], 1.0)):
        face = ya if s < 0 else yb
        for i in range(n + 1):
            x = x0 + (x1 - x0) * i / n
            if tag == "n" and GX0 - 0.020 < x < GX1 + 0.030:
                continue                      # under the gearbox's flange
            bolt(W, f"screw_beam_{tag}{i}", (x, face, z), "y",
                 r=0.0035, head=0.0025, sgn=s)


def rollers(spec, W, L):
    """Retaining caps on the rollers' END faces, and the screws that hold
    the bearing blocks down.

    The rollers are cylinders lying ACROSS the belt, so they extend along y
    and their ends face y.  A cap therefore has to sit beyond the end face
    (rule 2); putting it beside the barrel would bury it in the roller.

    The gap is 4 mm -- the bearing block's inner face is y 0.196 (south) /
    0.294 (north) and the roller's end face is 0.200 / 0.290 -- so the cap is
    exactly 4 mm thick and ends flush against the block.  There is no room
    in that gap for screw heads, which is why the three screws per roller
    stand on the bearing block's TOP face instead, where a housing's
    fasteners belong and where there is clear air above them.
    """
    R, HL, CONV_Z = L["ROLLER_R"], L["ROLLER_HL"], L["CONV_Z"]
    y_s, y_n = L["BELT_Y"] - HL, L["BELT_Y"] + HL        # the end faces
    yb_s, yb_n = L["RAIL_IN"][0] + 0.016, L["RAIL_IN"][1] - 0.016
    for end, xc in (("tail", L["BELT_TAIL"]), ("head", L["BELT_HALF_LEN"])):
        for side, yface, sgn, ybear in (("s", y_s, -1.0, yb_s),
                                        ("n", y_n, 1.0, yb_n)):
            half = 0.5 * abs(ybear - yface)               # fills the gap exactly
            _cyl(W, f"roller_cap_{end}{side}", (xc, yface, CONV_Z), "y",
                 R * 0.55, half, (0.58, 0.60, 0.63, 1.0), sgn=sgn)
            for k, dx in enumerate((-0.024, 0.0, 0.024)):
                bolt(W, f"bolt_bear_{end}{side}{k}",
                     (xc + dx, 0.5 * (yface + ybear), CONV_Z + 0.030), "z",
                     r=0.0045, head=0.0025)


# --------------------------------------------------------- control cabinet
def cabinet(spec, W, L):
    """E-stop, pilot lamp, hinges, latch and decals, all standing off the
    door's outer face.  Each part starts where the previous one ended, so
    the stack is flush and nothing is buried.
    """
    CAB_X, CAB_Y = L["CAB_X"], L["CAB_Y"]
    yf = CAB_Y[0] - 0.025            # the door's outer face
    stack = -1.0                     # everything grows outboard, -y

    # Emergency stop: plate, collar, mushroom.  x=-1.20 puts it clear of
    # the two pilot buttons build_cell already places at x=-1.27.
    # Each part starts where the PREVIOUS ONE'S OUTER FACE ended, which is
    # 2*half further out than the one before it -- _cyl spans pos..pos+2*half.
    _cyl(W, "enamel_estop_plate", (-1.20, yf, 0.72), "y", 0.030, 0.002,
         (0.95, 0.62, 0.06, 1.0), sgn=stack)              # -> yf - 0.004
    _cyl(W, "enamel_estop_collar", (-1.20, yf - 0.004, 0.72), "y", 0.020,
         0.006, (0.92, 0.92, 0.90, 1.0), sgn=stack)        # -> yf - 0.016
    _cyl(W, "enamel_estop_cap", (-1.20, yf - 0.016, 0.72), "y", 0.024, 0.008,
         (0.86, 0.13, 0.10, 1.0), sgn=stack)
    # pilot lamp with a bezel, below it
    _cyl(W, "enamel_lamp_bezel", (-1.20, yf, 0.60), "y", 0.021, 0.004,
         (0.55, 0.57, 0.60, 1.0), sgn=stack)              # -> yf - 0.008
    _cyl(W, "enamel_lamp_lens", (-1.20, yf - 0.008, 0.60), "y", 0.016, 0.005,
         (0.25, 0.85, 0.32, 1.0), sgn=stack)

    # hinges: barrels standing off the door's OUTER face, not in the 25 mm
    # gap behind it - the door is a solid box that fills that gap, so a
    # hinge in there would be buried inside it (rule 2).
    for k, hz in enumerate((0.16, L["CAB_H"] / 2, L["CAB_H"] - 0.16)):
        _cyl(W, f"cab_hinge_{k}", (CAB_X[0] + 0.041, yf, hz), "y", 0.011,
             0.0125, (0.66, 0.68, 0.71, 1.0), sgn=stack)
    _cyl(W, "cab_latch", (CAB_X[1] - 0.030, yf, 0.52), "y", 0.010, 0.008,
         (0.70, 0.72, 0.75, 1.0), sgn=stack)

    # Both labels go on the RAISED equipment plate (cv_cab_plate), which is
    # 7 mm proud of the door and spans z 0.40..0.56.  On the door instead
    # they land 1.2 mm inside that plate; above z 0.56 the door is already
    # occupied by the buttons (x -1.287..-1.253), the pilot lamp
    # (x -1.221..-1.179) and the e-stop (x -1.23..-1.17).
    ypl = CAB_Y[0] - 0.032                      # the equipment plate's face
    _box(W, "label_plate", (CAB_X[0] + 0.062, CAB_X[0] + 0.116),
         (ypl - 0.0012, ypl), (0.45, 0.50), rgba=(0.86, 0.87, 0.89, 1.0), **VIS)
    _box(W, "label_hazard", (CAB_X[0] + 0.130, CAB_X[0] + 0.210),
         (ypl - 0.0012, ypl), (0.40, 0.44), rgba=(1.0, 1.0, 1.0, 1.0), **VIS)
