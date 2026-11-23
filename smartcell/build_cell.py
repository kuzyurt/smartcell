"""Assembly-line work cell: verified SO-101 + official conveyor + factory table,
standing in a factory hall.

Assets (see assets/PROVENANCE.md for licences and every change made):
  * robot     mujoco_menagerie `robotstudio_so101` (Apache-2.0), unmodified
  * conveyor  google-deepmind/mujoco model/surfacevel/carousel.xml (Apache-2.0),
              official `geom/surfacevel` belt, cross-section adapted and
              lengthened 200 mm at the tail
  * table     vikashplus/furniture_sim simpleTable (Apache-2.0), joints fixed
  * hall      built in factory_hall.py (no open MJCF factory was found)
  * line      line_frame.py builds the machine around the belt: extruded side
              beams, tube legs on levelling feet, tie bars, the tail take-up,
              the guards and the flare guides
  * look      scene_look.py paints everything: procedural textures
              (make_textures.py -> assets/tex/), PBR materials and an
              8-light studio rig.  Physics is untouched by any of it.

On sourcing a ready-made assembly line: searched, and there is no openly
licensed MJCF conveyor or line section anywhere - the only belt in the whole
MuJoCo project is the carousel vendored here, and every public conveyor model
found (Sketchfab/GrabCAD/TurboSquid/Free3D and the icon packs) is licensed
CC-BY-NC or custom-licensed and ships as glTF/FBX/MAX, none of which MuJoCo
can load (it reads OBJ/STL/MSH, one material per mesh, no glTF).  A converted
static mesh would also have to carry the belt itself to look like a belt, and
the belt is the one surface that must stay a `geom/surfacevel` primitive: a
mesh belt would break the measured transport, the lane width and the grasp
clearances that everything else here is derived from.  So the line is: the
official surfacevel belt at its measured cross-section + a frame, drive unit,
controls and guards built here from the same layout constants, all aluminium.

Every placement below comes from a measurement, not from taste:
  * BASE_LIFT  the SO-101 base underside sits 2.4 mm *below* the `base` body
    origin (binary search on mj_geomDistance), so that is the lift used.
  * BELT_Y     top-down reach of the arm, mapped with IK at 2 cm resolution
    (arm_ik.py): the grasp window along a lane at y = +0.245 is
    x = 0.00 .. +0.20 and nothing beyond.  So the belt runs across the front
    of the robot, the infeed is at -X where the arm cannot reach (101 mm out
    of range), and the tote stands on the table inside the verified window.
  * TOTE_XY    a tote on the floor was unreachable (37-47 deg tilt, joints on
    their stops), so the tote stands on the table at (0.14, -0.175).

Layout, metres, +X right, +Y away from the viewer, +Z up:
    belt lane y=+0.245, 1.63 m of belt, belt top = table top = 0.625
    tail roller x=-0.94, head roller x=+0.74, frame x -1.065..+0.800
    robot (0,0) on the table, tote (0.02,-0.26) on the table in front of it
    parts ride +X from x=-0.72 to the head roller at x=+0.74, then drop into
    the discharge bin.
"""
import os

import mujoco
import mujoco_menagerie as mm

import conveyor_drive
import numpy as np

import factory_hall
import line_frame
import scene_look
from mjbox import box as _box, col

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")


# ------------------------------------------------------------------ heights
TABLE_TOP = 0.625                # top surface of the furniture_sim table
BASE_LIFT = 0.0024               # measured drop of the base underside
BASE_Z = TABLE_TOP + BASE_LIFT   # 0.6274

# ---------------------------------------------------------------- conveyor
# official conveyor section, see assets/conveyor.xml
CONV_X = 0.0                     # belt centre in x
BELT_Y = 0.245                   # lane centreline, the reachable offset
BELT_TOP = TABLE_TOP
BELT_HALF_LEN = 0.74             # HEAD roller centre - unchanged from upstream
BELT_TAIL = -0.94                # TAIL roller centre.  The belt was lengthened
                                 # 200 mm AT THE TAIL ONLY, so the discharge
                                 # end, the bin, the drive unit and the whole
                                 # measured pick window keep their numbers;
                                 # only the infeed run got longer.
BELT_HALF_W = 0.048              # belt half width
BELT_HALF_T = 0.025
CONV_Z = BELT_TOP - BELT_HALF_T  # frame origin height (roller axis height)
RAIL_Y = 0.070                   # rail centre offset: 140 mm clear lane
RAIL_HALF_T = 0.005
RAIL_HALF_H = 0.09
RAIL_CENTRE_Z = -0.005           # rail axis in the conveyor frame
BELT_SPEED = 0.10                 # indexing speed: the part must stay inside
                                   # the 160 mm grasp window for a full pick
ROLLER_R = 0.025                  # roller radius, from assets/conveyor.xml
ROLLER_HL = 0.045                 # roller half-length: the vendored rollers
                                   # are 90 mm long and run across the belt
CONV_X0 = BELT_TAIL - ROLLER_R   # -0.965: the tail roller's outer x
CONV_X1 = CONV_X + BELT_HALF_LEN + BELT_HALF_T   # 0.765: physical belt end

# ------------------------------------------------------------------- frame
# The line's superstructure is built in line_frame.py.  A real conveyor is two
# extruded side beams with the belt running between them and its underside on
# their tops, carried by legs that stand in the SAME 50 mm plane as the beams.
# So the machine occupies two 50 mm slabs of y - 0.120..0.170 and 0.320..0.370 -
# and the vendored side guides butt the beams' INNER faces, which is exactly the
# plane RAIL_OUT already named.  The frame's top is the belt's underside
# (BEAM_TOP = 0.575), which is why nothing in the redesign reaches above the
# belt inside the lane and the arm's verified clearance is untouched.
FRAME_X0 = -1.065                # west end of the beams (the tail take-up)
FRAME_X1 = 0.800                 # east end, past the head roller
FRAME_Y = ((0.120, 0.170), (0.320, 0.370))   # the two beams, 50 mm sections
FRAME_W = 0.050                  # beam section thickness in y
FRAME_SKIN = 0.012               # outer skin the T-slot is cut into
FRAME_GROOVE = 0.009             # T-slot depth (was 6 mm: too shallow to read
                                 # against the ribs beside it)
FRAME_RIB_Z = (0.478, 0.527)     # T-slot extent in z, on the outer face
FRAME_CAP = 0.012                # end-cap plate thickness
# 0.575, NOT BELT_TOP - BELT_HALF_T: CONV_Z is the belt's CENTRE, so its
# underside is CONV_Z - BELT_HALF_T.  Getting that wrong put the frame's top
# 25 mm inside the work table's top plate (verify_cell section 2 reported
# exactly -25.00 mm on cv_beam_s_rib1/table_top).
BEAM_TOP = CONV_Z - BELT_HALF_T           # 0.575: belt underside = frame top
BEAM_Z0 = 0.430                  # underside of the beams
CROSS_H = 0.040                  # tie bar, flush with the beams' bottom edge
LEG_X = ((-1.000, -0.950), (-0.650, -0.600), (-0.250, -0.200),
         (0.200, 0.250), (0.600, 0.650))
LEG_WALL = 0.008                 # leg tube wall
FOOT_Z = 0.085                   # top of the foot = bottom of the leg tube.
                                 # Raised from 70 mm with the stud and nut
                                 # scaled up so the foot reads as a levelling
                                 # foot and not as a plate on a stick.
GUARD_T = 0.008                  # guard sheet thickness
GUARD_H = 0.125                  # fence height above the frame top
GUARD_LIP = 0.022                # folded top lip, turned outboard
# The guard walls stand on the beams' top faces - but the near wall has to
# clear the work table, whose 1.20 x 0.75 m top reaches y = +0.150 and is at
# exactly the frame's height (z 0.575..0.625).  A wall anywhere inboard of
# 0.150 would run through the table's top plate over x -0.60..+0.60, which is
# what the outfeed guard did at -25.00 mm.  So the near wall sits at 0.156 -
# still on the beam (whose top face spans 0.120..0.170), 6 mm clear of the
# table's edge.  The beams themselves slide under the table's overhang, which
# is right: the table's top butts the conveyor's near side guide.
TABLE_EDGE_Y = -0.225 + 0.375                # +0.150: TABLE_POS.y + half
GUARD_Y0 = TABLE_EDGE_Y + 0.006              # 0.156
GUARD_X = (("in", (-0.80, -0.62)), ("out", (0.54, 0.68)))
RAIL_OUT = (BELT_Y - RAIL_Y - RAIL_HALF_T,   # 0.170 / 0.320 = beam inner faces
            BELT_Y + RAIL_Y + RAIL_HALF_T)
RAIL_IN = (BELT_Y - RAIL_Y + RAIL_HALF_T,    # 0.180 / 0.310 = guide inner faces
           BELT_Y + RAIL_Y - RAIL_HALF_T)
# ACCESS WINDOW in front of the robot.  The vendored guides are interrupted at
# the pick station (x -0.02..+0.22) so the wrist can come down; nothing the
# redesign added stands above the belt surface anywhere in here either, and the
# frame itself tops out 50 mm BELOW the belt top, so the window the arm works
# in is now clear at every height rather than only above the belt.
PICK_OPEN = (CONV_X - 0.16, CONV_X + 0.36)
RAIL_TOP = CONV_Z + RAIL_CENTRE_Z + RAIL_HALF_H   # 0.685, top of the guides
BEAR_HX = 0.040                              # bearing block half length

# ---------------------------------------------------------------- barriers
# The line's side guides and fence panels are taken OFF (user request: the arm
# needs the lane's full cross-section, not a 140 mm slot between two 60 mm
# rails).  They are built exactly as verified and then deleted from the spec by
# name, so line_frame.py and the vendored conveyor.xml stay untouched and the
# whole layout is still driven by the constants above.  Set BARRIERS = True to
# rebuild the ds_0001 cell.
BARRIERS = False
BARRIER_GEOMS = (
    "rail_tail", "rail_head", "rail_tail_far", "rail_head_far",
    "cv_guard_in_s", "cv_guard_in_n", "cv_guard_in_lip_s", "cv_guard_in_lip_n",
    "label_guard_in",
    "cv_guard_out_s", "cv_guard_out_n", "cv_guard_out_lip_s", "cv_guard_out_lip_n",
    "label_guard_out",
    "cv_sensor_0", "cv_led_0",          # the photo-eye stands on the far guide
)


def _strip_barriers(spec):
    """Delete the side guides, fence panels and the rail-mounted photo-eye."""
    if BARRIERS:
        return 0
    drop = [g for g in list(spec.geoms) if g.name in BARRIER_GEOMS]
    for g in drop:
        spec.delete(g)
    return len(drop)
CAB_X, CAB_Y = (-1.42, -1.12), (-0.34, -0.06)   # control cabinet footprint
CAB_H = 0.86

# ------------------------------------------------------------------- table
TABLE_POS = (0.0, -0.225, 0.0)  # puts the robot 150 mm from the back edge
TABLE_X = (-0.60, 0.60)
TABLE_Y = (-0.60, 0.15)

# -------------------------------------------------------------------- tote
# directly beside the arm on the table's front edge.  Measured with the IK
# sweep: at the release height the arm can only place top-down in a narrow
# band around (0.14..0.18, -0.17) on this side - closer to the arm or more
# central is out of reach for this 5-DoF wrist (checked in verify_cell).
TOTE_X = (0.040, 0.240)
TOTE_Y = (-0.265, -0.085)
TOTE_WALL = 0.010
TOTE_FLOOR = TABLE_TOP
TOTE_RIM = TABLE_TOP + 0.090
TOTE_INNER = (TOTE_X[0] + TOTE_WALL, TOTE_X[1] - TOTE_WALL,
              TOTE_Y[0] + TOTE_WALL, TOTE_Y[1] - TOTE_WALL)
TOTE_XY = (0.5 * (TOTE_X[0] + TOTE_X[1]), 0.5 * (TOTE_Y[0] + TOTE_Y[1]))
TOTE_RELEASE_Z = TABLE_TOP + 0.050

# ----------------------------------------------------------- discharge bin
BIN_X = (0.70, 0.98)
BIN_Y = (0.165, 0.325)
BIN_WALL = 0.012
BIN_RIM = 0.30
BIN_INNER = (BIN_X[0] + BIN_WALL, BIN_X[1] - BIN_WALL,
             BIN_Y[0] + BIN_WALL, BIN_Y[1] - BIN_WALL)

# ------------------------------------------------------------------- parts
CUBE = 0.032
CUBE_MASS = 0.08
POOL = 5
SPAWN_X = -0.72                  # on the belt, under the infeed guard and out
                                 # of the arm's reach; it is the flare mouth,
                                 # exactly where the lane funnel starts
BELT_END_X = CONV_X1 - 0.03      # part leaves the belt over the head roller
DESPAWN_Z = 0.24                 # below the bin rim: gone from play, hidden
PICK_X = (0.02, 0.16)            # top-down window with a real joint margin
HOME_XYZ = (0.26, 0.18, 0.80)                    # free pose, clear of the cell, 0.00 mm IK error

# infeed pallet: inactive parts wait here, so the spawn has a visible source.
# Moved 170 mm west when the frame grew: its east block used to stop 20 mm
# short of the westmost foot plate, and the new frame's foot plate reaches
# into that gap (measured overlap before the move: 0.070 x 0.290 m in plan,
# 8 mm in z - a real one).
TRAY_X = (-1.62, -1.12)
TRAY_Y = (0.10, 0.42)
TRAY_H = 0.06
PARK = (-1.57, 0.26, TRAY_H)
PARK_DX = 0.10

CUBE_COLORS = [
    (0.90, 0.25, 0.22, 1.0),
    (0.20, 0.72, 0.35, 1.0),
    (0.25, 0.50, 0.90, 1.0),
    (0.95, 0.65, 0.15, 1.0),
    (0.60, 0.35, 0.85, 1.0),
]

C_TOTE = (0.14, 0.34, 0.60, 1.0)
C_TOTE_IN = (0.09, 0.25, 0.46, 1.0)
C_BIN = (0.24, 0.40, 0.30, 1.0)
C_STAND = (0.30, 0.32, 0.35, 1.0)
C_GUARD = (0.90, 0.55, 0.10, 1.0)
C_MOTOR = (0.55, 0.58, 0.62, 1.0)
# NOTE: colours are historical fallback values.  The compiled look comes from
# scene_look.py (materials from assets/tex textures + studio lighting); these
# constants only matter if scene_look were removed.


def _look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """Camera position + quaternion aimed at target (a MuJoCo camera looks -Z)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    zc = -f
    xc = np.cross(np.asarray(up, float), zc)
    if np.linalg.norm(xc) < 1e-6:
        # looking straight up or down: world up is parallel to the view axis,
        # so pick any axis that is not, or the basis comes out as NaN
        alt = np.array([1.0, 0.0, 0.0]) if abs(zc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        xc = np.cross(alt, zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    r = np.column_stack((xc, yc, zc))
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
             (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s,
             (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s,
             (r[1, 2] + r[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
             (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return pos, q


def _attach(spec, name, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    """Insert a vendored asset file into the world body at a given pose."""
    child = mujoco.MjSpec.from_file(os.path.join(ASSETS, name))
    frame = spec.worldbody.add_frame()
    frame.pos = col(pos)
    frame.quat = col(quat)
    spec.attach(child, frame=frame)
    for g in spec.geoms:            # attach prefixes names with the frame path
        if g.name and "/" in g.name:
            g.name = g.name.rsplit("/", 1)[-1]
    return frame


def build_spec():
    spec = mm.get("robotstudio_so101").spec()
    spec.body("base").pos = col([0.0, 0.0, BASE_Z])
    W = spec.worldbody

    # see the note on the grasp site below
    gs = spec.body("gripper").add_site()
    gs.name = "grasp"
    gs.pos = col([0.01948, 0.0118, -0.0993])
    gs.size = col([0.004, 0.004, 0.004])
    gs.rgba = col([1.0, 0.0, 1.0, 0.3])

    factory_hall.build_hall(spec, cell_box=(-1.05, 1.10, -0.75, 0.50))
    factory_hall.add_props(spec, {"hx": factory_hall.HX, "hy": factory_hall.HY})

    # set dressing: column pads and bollards, no physics role
    scene_look.add_props(spec)

    # ------------------------------------------------------------- the frame
    # One dict, two builders: the layout is passed by value so build_cell.py
    # stays the only source of truth for every coordinate on the line.
    layout = dict(
        CONV_X=CONV_X, CONV_Z=CONV_Z, ROLLER_R=ROLLER_R, BELT_Y=BELT_Y,
        BELT_TOP=BELT_TOP, BELT_TAIL=BELT_TAIL, BELT_HALF_T=BELT_HALF_T,
        BELT_HALF_LEN=BELT_HALF_LEN, CONV_X1=CONV_X1,
        RAIL_Y=RAIL_Y, RAIL_HALF_T=RAIL_HALF_T, RAIL_OUT=RAIL_OUT,
        RAIL_IN=RAIL_IN, RAIL_TOP=RAIL_TOP, BEAR_HX=BEAR_HX,
        FRAME_X0=FRAME_X0, FRAME_X1=FRAME_X1, FRAME_Y=FRAME_Y,
        FRAME_W=FRAME_W, FRAME_SKIN=FRAME_SKIN, FRAME_GROOVE=FRAME_GROOVE,
        FRAME_RIB_Z=FRAME_RIB_Z, FRAME_CAP=FRAME_CAP,
        BEAM_TOP=BEAM_TOP, BEAM_Z0=BEAM_Z0, CROSS_H=CROSS_H,
        LEG_X=LEG_X, LEG_WALL=LEG_WALL, FOOT_Z=FOOT_Z,
        GUARD_T=GUARD_T, GUARD_H=GUARD_H, GUARD_LIP=GUARD_LIP,
        GUARD_X=GUARD_X, GUARD_Y0=GUARD_Y0, PICK_OPEN=PICK_OPEN,
        CAB_X=CAB_X, CAB_Y=CAB_Y, CAB_H=CAB_H, ROLLER_HL=ROLLER_HL)

    # the machine itself: extruded side beams, tube legs on levelling feet,
    # cross ties, the tail take-up, the guards, the flared infeed guides
    line_frame.build(spec, W, layout)

    # ------------------------------------------ mechanical detail (visual only)
    # gearbox, coupling, motor fins, every bolt and socket screw.  All of it
    # contype=0 conaffinity=0, so none of it can touch the physics.
    conveyor_drive.build(spec, W, layout)

    # ------------------------------------------------- work table (vendored)
    _attach(spec, "table.xml", TABLE_POS)

    # -------------------------------------------- conveyor (vendored, official)
    _attach(spec, "conveyor.xml", (CONV_X, BELT_Y, CONV_Z))

    # --------------------------------------------------- drive unit (head end)
    # The gearbox is bolted to the FAR beam's outer face, the motor cylinder
    # hangs off its flange and the terminal box sits on top: the drive train
    # lives entirely outside the belt's envelope.  conveyor_drive owns these
    # numbers (GY0/GY1/GYC/MX) and draws the shaft, the coupling, the ribs,
    # the sight glass and the nameplate that go with them.
    GY0, GY1, GYC = conveyor_drive.GY0, conveyor_drive.GY1, conveyor_drive.GYC
    _box(W, "cv_gear_0", (0.70, 0.80), (GY0, GY1), (0.545, 0.660),
         rgba=C_MOTOR)
    m = W.add_geom()
    m.name = "cv_motor"
    m.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    m.size = col([0.040, 0.070, 0.0])
    # 66 mm of clear air between the gearbox face and the motor, so a flexible
    # coupling fits between them - without it the two castings butt together
    # and the drive reads as one lump
    m.pos = col([conveyor_drive.MX, GYC, CONV_Z])
    m.quat = col([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # along x
    m.rgba = col(list(C_MOTOR))
    _box(W, "cv_term_0", (0.72, 0.78), (GYC - 0.014, GYC + 0.014),
         (0.660, 0.700), rgba=C_MOTOR)

    # --------------------------------------------------- control cabinet (tail)
    _box(W, "cv_cab_body", CAB_X, CAB_Y, (0.0, CAB_H), rgba=C_MOTOR)
    _box(W, "cv_cab_door", (CAB_X[0] + 0.03, CAB_X[1] - 0.03),
         (CAB_Y[0] - 0.025, CAB_Y[0]), (0.06, CAB_H - 0.06), rgba=C_MOTOR)
    _box(W, "cv_cab_plate", (CAB_X[0] + 0.06, CAB_X[1] - 0.06),
         (CAB_Y[0] - 0.032, CAB_Y[0] - 0.025), (0.40, 0.56), rgba=C_MOTOR)
    for k, (zc, colr) in enumerate(((0.72, (0.20, 0.85, 0.30, 1.0)),
                                    (0.64, (0.90, 0.20, 0.16, 1.0)))):
        b = W.add_geom()
        b.name = f"cv_btn_{k}"
        b.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        b.size = col([0.017, 0.009, 0.0])
        # shaft along y, flat face exactly on the door's face
        b.pos = col([-1.27, CAB_Y[0] - 0.025 - 0.009, zc])
        b.quat = col([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        b.rgba = col(list(colr))
    # conduit: cabinet -> across the floor -> into the near beam's outer face.
    # The old run entered the flat side panel; the beam's outer face at this
    # height is the extrusion's upper rib, so the conduit's end lands on it and
    # the run is carried at both ends instead of hanging in the air.
    _box(W, "cv_conduit_0", (CAB_X[1], FRAME_X0), (-0.200, -0.170),
         (0.500, 0.540), rgba=C_STAND)
    _box(W, "cv_conduit_1", (FRAME_X0, FRAME_X0 + 0.020),
         (-0.185, FRAME_Y[0][0]), (0.500, 0.540), rgba=C_STAND)

    # --------------------------------------- pick-station photo-eye on the rail
    # a bracket standing on the far guide with an LED facing across the lane.
    # It sits just past the access opening (PICK_OPEN[1]), so the window the
    # robot works in stays completely clear of hardware
    _box(W, "cv_sensor_0", (PICK_OPEN[1] + 0.01, PICK_OPEN[1] + 0.09),
         (BELT_Y + RAIL_Y - 0.010, BELT_Y + RAIL_Y + 0.010),
         (RAIL_TOP, RAIL_TOP + 0.056), rgba=C_MOTOR)
    _box(W, "cv_led_0", (PICK_OPEN[1] + 0.03, PICK_OPEN[1] + 0.07),
         (BELT_Y + RAIL_Y - 0.014, BELT_Y + RAIL_Y - 0.010),
         (RAIL_TOP + 0.014, RAIL_TOP + 0.042), rgba=(0.95, 0.30, 0.20, 1.0))

    # ------------------------------------------------------------------ tote
    _box(W, "tote_floor", TOTE_INNER[:2], TOTE_INNER[2:],
         (TOTE_FLOOR, TOTE_FLOOR + TOTE_WALL), rgba=C_TOTE_IN)
    # Box-bottom landmark (VISUAL ONLY: contype/conaffinity 0, no mass, no
    # friction - physics is untouched): a 2mm hazard-stripe inlay flush on
    # the tote floor. The policy input is 96x96 GRAY, so the box needs a
    # LUMINANCE pattern, not a color: yellow/black stripes stay bright/dark
    # after the gray mean. Named boxmark_* (NOT tote_*) so scene_look gives
    # it the hazard material instead of tote-blue plastic. It moves with
    # the per-episode tote offset (see collect_state.apply_tote/TOTE_GEOMS).
    # Parts rest 2mm higher; _in_tote still holds (floor < z < rim).
    _box(W, "boxmark_floor", TOTE_INNER[:2], TOTE_INNER[2:],
         (TOTE_FLOOR + TOTE_WALL, TOTE_FLOOR + TOTE_WALL + 0.002),
         rgba=(1.0, 1.0, 1.0, 1.0), contype=0, conaffinity=0)
    for tag, yr in (("n", (TOTE_Y[0], TOTE_Y[0] + TOTE_WALL)),
                    ("s", (TOTE_Y[1] - TOTE_WALL, TOTE_Y[1]))):
        _box(W, f"tote_wall_{tag}", TOTE_X, yr, (TOTE_FLOOR, TOTE_RIM - 0.012),
             rgba=C_TOTE)
    for tag, xr in (("w", (TOTE_X[0], TOTE_X[0] + TOTE_WALL)),
                    ("e", (TOTE_X[1] - TOTE_WALL, TOTE_X[1]))):
        _box(W, f"tote_wall_{tag}", xr,
             (TOTE_Y[0] + TOTE_WALL, TOTE_Y[1] - TOTE_WALL),
             (TOTE_FLOOR, TOTE_RIM - 0.012), rgba=C_TOTE)
    lip = 0.008
    for tag, yr in (("n", (TOTE_Y[0] - lip, TOTE_Y[0] + TOTE_WALL)),
                    ("s", (TOTE_Y[1] - TOTE_WALL, TOTE_Y[1] + lip))):
        _box(W, f"tote_lip_{tag}", (TOTE_X[0] - lip, TOTE_X[1] + lip), yr,
             (TOTE_RIM - 0.012, TOTE_RIM), rgba=C_TOTE)
    for tag, xr in (("w", (TOTE_X[0] - lip, TOTE_X[0] + TOTE_WALL)),
                    ("e", (TOTE_X[1] - TOTE_WALL, TOTE_X[1] + lip))):
        _box(W, f"tote_lip_{tag}", xr,
             (TOTE_Y[0] + TOTE_WALL, TOTE_Y[1] - TOTE_WALL),
             (TOTE_RIM - 0.012, TOTE_RIM), rgba=C_TOTE)

    # --------------------------------------------------------- discharge bin
    _box(W, "bin_floor", BIN_INNER[:2], BIN_INNER[2:], (0.0, BIN_WALL),
         rgba=C_BIN)
    for tag, yr in (("n", (BIN_Y[0], BIN_Y[0] + BIN_WALL)),
                    ("s", (BIN_Y[1] - BIN_WALL, BIN_Y[1]))):
        _box(W, f"bin_wall_{tag}", BIN_X, yr, (0.0, BIN_RIM - 0.012), rgba=C_BIN)
    for tag, xr in (("w", (BIN_X[0], BIN_X[0] + BIN_WALL)),
                    ("e", (BIN_X[1] - BIN_WALL, BIN_X[1]))):
        _box(W, f"bin_wall_{tag}", xr, (BIN_Y[0] + BIN_WALL, BIN_Y[1] - BIN_WALL),
             (0.0, BIN_RIM - 0.012), rgba=C_BIN)
    for tag, yr in (("n", (BIN_Y[0] - lip, BIN_Y[0] + BIN_WALL)),
                    ("s", (BIN_Y[1] - BIN_WALL, BIN_Y[1] + lip))):
        _box(W, f"bin_lip_{tag}", (BIN_X[0] - lip, BIN_X[1] + lip), yr,
             (BIN_RIM - 0.012, BIN_RIM), rgba=C_BIN)
    for tag, xr in (("w", (BIN_X[0] - lip, BIN_X[0] + BIN_WALL)),
                    ("e", (BIN_X[1] - BIN_WALL, BIN_X[1] + lip))):
        _box(W, f"bin_lip_{tag}", xr, (BIN_Y[0] + BIN_WALL, BIN_Y[1] - BIN_WALL),
             (BIN_RIM - 0.012, BIN_RIM), rgba=C_BIN)

    # -------------------------------------------------------- infeed pallet
    # 3 slats, 80 mm wide, 40 mm apart, exactly filling the 320 mm tray.
    # The pitch used to be 40 mm with the same 80 mm width, which made the
    # slats overlap each other by 40 mm - invisible in the old gate, which
    # was reading a wrong distance for overlapping boxes.
    for i in range(3):
        _box(W, f"infeed_slat_{i}", TRAY_X,
             (TRAY_Y[0] + 0.12 * i, TRAY_Y[0] + 0.12 * i + 0.08),
             (TRAY_H - 0.02, TRAY_H), rgba=(0.58, 0.47, 0.32, 1.0))
    for i, xr in enumerate(((TRAY_X[0], TRAY_X[0] + 0.08),
                            (0.5 * (TRAY_X[0] + TRAY_X[1]) - 0.04,
                             0.5 * (TRAY_X[0] + TRAY_X[1]) + 0.04),
                            (TRAY_X[1] - 0.08, TRAY_X[1]))):
        _box(W, f"infeed_block_{i}", xr, TRAY_Y, (0.0, TRAY_H - 0.02),
             rgba=(0.48, 0.38, 0.26, 1.0))

    # ------------------------------------------------------------ part pool
    for i in range(POOL):
        b = W.add_body()
        b.name = f"cube_{i}"
        b.pos = col([PARK[0] + i * PARK_DX, PARK[1], PARK[2] + CUBE / 2])
        b.add_freejoint()
        g = b.add_geom()
        g.name = f"cube_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = col([CUBE / 2] * 3)
        g.mass = CUBE_MASS
        g.rgba = col(list(CUBE_COLORS[i]))
        g.friction = col([1.2, 0.005, 0.0001])
        g.solref = col([0.003, 1.0])

    # ------------------------------------------------------------- cameras
    # Framing: the cell runs from the infeed pallet at x=-1.45 to the bin at
    # x=+0.98, and from the table's front edge y=-0.60 to the tray at y=+0.42.
    c = W.add_camera()
    c.name = "overhead"
    p, q = _look_at((-0.20, -0.75, 2.35), (-0.20, -0.05, 0.60))
    c.pos = col(p)
    c.quat = col(q)
    c.fovy = 56.0

    c = W.add_camera()
    c.name = "preview"
    p, q = _look_at((-1.55, -1.55, 1.55), (-0.10, 0.05, 0.66))
    c.pos = col(p)
    c.quat = col(q)
    c.fovy = 52.0

    c = W.add_camera()
    c.name = "hall"
    p, q = _look_at((2.15, -1.95, 1.55), (-0.25, 0.0, 0.72))
    c.pos = col(p)
    c.quat = col(q)
    c.fovy = 60.0

    # look: materials from textures, per-geom square texture tiles, studio
    # lighting, PBR finish
    scene_look._add_materials(spec)
    scene_look.paint(spec)
    scene_look._fit_textures(spec)
    scene_look.relight(spec)
    n = _strip_barriers(spec)
    if n:
        print(f"barriers: removed {n} geoms ({'on' if BARRIERS else 'off'})")
    return spec


def build_model():
    m = build_spec().compile()
    return scene_look.polish(m)


if __name__ == "__main__":
    m = build_model()
    print(f"cell ok: nq={m.nq} nu={m.nu} ncam={m.ncam} ngeom={m.ngeom} "
          f"nmat={m.nmat} nlight={m.nlight}")







