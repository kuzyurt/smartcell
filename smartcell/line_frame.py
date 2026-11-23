"""Structure of the assembly line: the frame the belt runs in.

The belt itself, its two side guides and its two end rollers are the vendored
`surfacevel` model (assets/conveyor.xml) and none of them may be moved: the
belt's transport, the 140 mm lane width and the robot's grasp window are all
measured off those five geoms.  Everything AROUND them is built here.

This file replaces the first attempt - eight box legs and two flat 8 mm side
plates - which read as boxes because that is exactly what they were.  What a
real line looks like, and what is therefore modelled:

  side beams   two aluminium extrusions, 50 wide x 145 tall, one either side of
               the lane, running the whole length.  Their TOP face is the
               belt's underside (z = 0.575) and the vendored side guides bolt
               straight onto their inner face, so the guide rises out of the
               top of the frame exactly as it does on a real machine.  Each
               beam carries a T-slot groove down its outer face - the one
               feature that says "extrusion" and not "box".
  legs         square tube, 50 x 50, in the SAME 50 mm plane as the beams they
               carry, so the load path is a straight line from the belt to the
               floor instead of stepping 50 mm inboard as it used to.  Each
               tube stops 12 mm short of the beam it carries, because a
               bolted bracket goes in that gap.
  brackets     a plate under every beam at every leg station, oversailing the
               tube 30 mm each way, with two bolt heads under it.  This is
               what stops the junction reading as two boxes touched
               together.
  slot screws  socket heads sitting in the beams' T-slot, two per leg
               station, recessed 3 mm inside the outer face.  Nothing else on
               the profile says how big the machine is at a glance.
  feet         pad -> stud -> hex nut -> stud -> tube.  A line stands on
               levelling feet because no slab is flat; it is also why the leg
               tube stops 70 mm short of the concrete.
  cross members  tie the two beams together at every leg station.
  take-up      at the tail: a cross plate carrying two threaded take-up screws
               that pull the tail roller's bearing blocks, with locknuts.  This
               is how belt tension is actually set.
  guards       sheet tunnels over the infeed and the outfeed, standing on the
               beam tops.
  flares       the funnel that centres a part before the pick station.  These
               are the only parts in this file that touch a moving part: they
               are colliding geoms on purpose, and they keep their measured
               geometry (the plate's inner corner lands on the rail's inner
               face and the window is 7 mm above the belt).

Geometry rules, both of them enforced by verify_cell.py sections 2 and 3:

  * NOTHING OVERLAPS.  Every profile here is decomposed as a PARTITION - beams
    into core + two ribs + slot bottom, legs into four plates - so neighbouring
    pieces share faces exactly and no piece is inside another.  Detail butts
    against the surface it is fixed to.
  * NOTHING FLOATS.  Every geom touches the floor or a neighbour.  That is why
    the take-up screws run from the bearing block to the tail plate instead of
    sticking out into the air, why the guard walls stand on the beam tops
    instead of hanging beside them, and why the foot stack is five pieces that
    each sit on the one below.

And one rule that is about the robot rather than about looks: NOTHING ADDED
HERE RISES ABOVE z = 0.575 INSIDE THE LANE.  0.575 is the belt's underside and
the top of the frame; the only things above it are the vendored side guides
(which the original model already interrupts at the pick station for a measured
reason) and the guard tunnels, which are metres out of the arm's reach.  The
arm's clearance is therefore the clearance that was already verified, and this
file cannot invalidate it.

Layout comes in as a dict from build_cell.py, so there is one source of truth
for every coordinate.
"""
import mujoco
import numpy as np

from mjbox import box as _box, col

# Structural geoms DO collide (contype 1, the _box default): the frame is
# solid, and the arm is designed never to touch it.  Fasteners added by
# conveyor_drive.py are visual only.
C_STEEL = (0.30, 0.32, 0.35, 1.0)
C_PLATE = (0.46, 0.48, 0.51, 1.0)
# Safety yellow.  Brighter than a literal orange because these fences are
# VERTICAL surfaces: they see the rig at a grazing angle and measured 76/255
# rendered where a flat floor marking of the same colour measures 176.
C_GUARD = (0.97, 0.72, 0.10, 1.0)


def _way(outer, s, a, b):
    """y range of a band lying `a`..`b` metres INWARD of the outer face.

    `s` is the inward direction on that side (+1 for the -y beam, -1 for the
    +y one).  Working in "depth inward" is what keeps the two sides exact
    mirrors of each other without duplicating every number.
    """
    p, q = outer + s * a, outer + s * b
    return (min(p, q), max(p, q))


def _cyl(W, name, pos, axis, r, half, rgba, sgn=1.0):
    """Cylinder growing away from the face point `pos`, along `axis` * sgn."""
    v = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
         "z": np.array([0, 0, 1.0])}[axis]
    q = {"x": np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0.0]),
         "y": np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]),
         "z": np.array([1.0, 0, 0, 0.0])}[axis]
    g = W.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = col([r, half, 0.0])
    g.pos = col(np.asarray(pos, float) + v * sgn * half)
    g.quat = col(q)
    g.rgba = col(list(rgba))
    return g


def _tube(W, name, xr, yr, zr, t, rgba):
    """Hollow square tube: four plates that share edges, so no overlap.

    Each plate is a full side; the two side plates span only the clear width,
    so they butt the flat plates on a face instead of crossing them.  A solid
    box would have been one geom, but a real leg is a tube, and at the foot the
    open end is what the levelling stud screws into.
    """
    x0, x1 = xr
    y0, y1 = yr
    _box(W, f"{name}_s", (x0, x1), (y0, y0 + t), zr, rgba=rgba)
    _box(W, f"{name}_n", (x0, x1), (y1 - t, y1), zr, rgba=rgba)
    _box(W, f"{name}_w", (x0, x0 + t), (y0 + t, y1 - t), zr, rgba=rgba)
    _box(W, f"{name}_e", (x1 - t, x1), (y0 + t, y1 - t), zr, rgba=rgba)


def build(spec, W, L):
    n0 = len(spec.geoms)
    beams(spec, W, L)
    slot_screws(spec, W, L)
    legs(spec, W, L)
    brackets(spec, W, L)
    feet(spec, W, L)
    cross(spec, W, L)
    bearings(spec, W, L)
    takeup(spec, W, L)
    guards(spec, W, L)
    flares(spec, W, L)
    return len(spec.geoms) - n0


# ------------------------------------------------------------- side beams
def beams(spec, W, L):
    """Two extruded side beams, each with a T-slot down its outer face.

    The section is a partition: a 38 mm core covers the whole height, a 12 mm
    skin covers the outer 12 mm of the height EXCEPT the 41 mm the slot opens
    into, and the slot's own 6 mm bottom fills that opening.  So the outer face
    reads as rib / recessed slot / rib, which is what an aluminium profile
    looks like, and no piece overlaps another.
    """
    x0, x1 = L["FRAME_X0"], L["FRAME_X1"]
    cap = L["FRAME_CAP"]
    z0, z1 = L["BEAM_Z0"], L["BEAM_TOP"]
    g0, g1 = L["FRAME_RIB_Z"]            # the slot's extent in z
    skin, deep = L["FRAME_SKIN"], L["FRAME_GROOVE"]
    width = L["FRAME_W"]                                 # 0.050 section
    for tag, (y0, y1) in (("s", L["FRAME_Y"][0]), ("n", L["FRAME_Y"][1])):
        s = 1.0 if y0 < 0.245 else -1.0                  # inward direction
        outer = y0 if s > 0 else y1
        inner = y1 if s > 0 else y0
        body = (x0 + cap, x1 - cap)
        _box(W, f"cv_beam_{tag}_core", body, _way(outer, s, skin, width),
             (z0, z1), rgba=(0.50, 0.52, 0.56, 1.0))
        _box(W, f"cv_beam_{tag}_rib0", body, _way(outer, s, 0.0, skin),
             (z0, g0), rgba=(0.50, 0.52, 0.56, 1.0))
        _box(W, f"cv_beam_{tag}_rib1", body, _way(outer, s, 0.0, skin),
             (g1, z1), rgba=(0.50, 0.52, 0.56, 1.0))
        _box(W, f"cv_beam_{tag}_slot", body, _way(outer, s, deep, skin),
             (g0, g1), rgba=(0.42, 0.44, 0.48, 1.0))
        # end caps close the profile at both ends (the slot would otherwise be
        # open to the inside of the tube, which is the giveaway of a fake)
        for k, xr in enumerate(((x0, x0 + cap), (x1 - cap, x1))):
            _box(W, f"cv_beam_{tag}_cap{k}", xr, (min(outer, inner),
                                                  max(outer, inner)),
                 (z0, z1), rgba=(0.44, 0.46, 0.50, 1.0))


# ------------------------------------------------------------------- legs
LEG_BRACKET = 0.012              # plate between the leg top and the beam
LEG_OVER = 0.030                 # how far it oversails the tube each side


def legs(spec, W, L):
    """Square-tube legs, one pair per station, under the beams.

    The tube stops at the bracket, not at the beam: legs() and brackets()
    partition the same 12 mm, so neither one penetrates the other and neither
    one floats.
    """
    z0, z1 = L["FOOT_Z"], L["BEAM_Z0"] - LEG_BRACKET
    t = L["LEG_WALL"]
    for i, (x0, x1) in enumerate(L["LEG_X"]):
        for j, (y0, y1) in enumerate(L["FRAME_Y"]):
            _tube(W, f"cv_leg_{i}{j}", (x0, x1), (y0, y1), (z0, z1), t,
                  (0.42, 0.44, 0.48, 1.0))


def brackets(spec, W, L):
    """The bolted plate that joins each leg top to the beam above it.

    Flush with the beam's 50 mm section, so the pair still reads as one leg
    line from the side, and two bolt heads hang under it either side of the
    tube - the fasteners are the whole reason the joint stops looking like two
    boxes stacked on each other.
    """
    z1 = L["BEAM_Z0"]
    for i, (x0, x1) in enumerate(L["LEG_X"]):
        for j, (y0, y1) in enumerate(L["FRAME_Y"]):
            _box(W, f"cv_bracket_{i}{j}",
                 (x0 - LEG_OVER, x1 + LEG_OVER), (y0, y1),
                 (z1 - LEG_BRACKET, z1), rgba=(0.42, 0.44, 0.48, 1.0))
            cy = 0.5 * (y0 + y1)
            for k, bx in ((0, x0 - LEG_OVER / 2), (1, x1 + LEG_OVER / 2)):
                _cyl(W, f"bolt_bracket_{i}{j}{k}", (bx, cy, z1 - LEG_BRACKET),
                     "z", 0.005, 0.004, (0.86, 0.87, 0.89, 1.0), sgn=-1.0)


def slot_screws(spec, W, L):
    """Socket heads in the beams' T-slot, one pair per leg station.

    They sit on the slot's floor and stop 3 mm inside the outer face, so the
    profile's silhouette is unchanged - nothing here can reach the arm.
    """
    g0, g1 = L["FRAME_RIB_Z"]
    zc = 0.5 * (g0 + g1)
    for tag, (y0, y1) in (("s", L["FRAME_Y"][0]), ("n", L["FRAME_Y"][1])):
        s = 1.0 if y0 < 0.245 else -1.0
        floor = (y0 if s > 0 else y1) + s * L["FRAME_GROOVE"]
        for i, (x0, x1) in enumerate(L["LEG_X"]):
            xc = 0.5 * (x0 + x1)
            for k, dx in ((0, -0.025), (1, 0.025)):
                _cyl(W, f"bolt_slot_{tag}{i}{k}", (xc + dx, floor, zc), "y",
                     0.006, 0.003, (0.86, 0.87, 0.89, 1.0), sgn=-s)


def feet(spec, W, L):
    """Levelling feet: pad, stud, hex nut, stud, into the open leg tube.

    Five pieces that each rest on the one below.  The upper stud ends exactly
    at the tube's lower end, so it reads as screwing up into the leg (which is
    how these are fitted) and - because the tube is hollow - the stud's top
    disc never touches the tube's walls, only the nut beneath it.  Nothing is
    buried and nothing floats.
    """
    fz = L["FOOT_Z"]
    pad_t, pad_margin = 0.008, 0.013          # 76 mm pad on a 50 mm tube
    stud_r, nut_r = 0.010, 0.017
    nut0, nut1 = 0.046, 0.064
    for i, (x0, x1) in enumerate(L["LEG_X"]):
        cx = 0.5 * (x0 + x1)
        for j, (y0, y1) in enumerate(L["FRAME_Y"]):
            cy = 0.5 * (y0 + y1)
            _box(W, f"cv_foot_plate_{i}{j}",
                 (x0 - pad_margin, x1 + pad_margin),
                 (y0 - pad_margin, y1 + pad_margin), (0.0, pad_t),
                 rgba=(0.58, 0.60, 0.63, 1.0))
            _cyl(W, f"cv_foot_stud_a{i}{j}", (cx, cy, pad_t), "z", stud_r,
                 (nut0 - pad_t) / 2, (0.86, 0.87, 0.89, 1.0))
            _cyl(W, f"cv_foot_nut_{i}{j}", (cx, cy, nut0), "z", nut_r,
                 (nut1 - nut0) / 2, (0.30, 0.31, 0.34, 1.0))
            _cyl(W, f"cv_foot_stud_b{i}{j}", (cx, cy, nut1), "z", stud_r,
                 (fz - nut1) / 2, (0.86, 0.87, 0.89, 1.0))
            for k, by in ((0, y0 - pad_margin / 2), (1, y1 + pad_margin / 2)):
                _cyl(W, f"bolt_anchor_{i}{j}{k}",
                     (cx, by, pad_t), "z", 0.005, 0.004,
                     (0.86, 0.87, 0.89, 1.0))


# --------------------------------------------------------- cross members
def cross(spec, W, L):
    """Tie bars between the two beams at every leg station.

    A single solid box works here - unlike the legs it is not a tube - and its
    two end faces land exactly on the beams' inner faces, which is also the
    plane the vendored side guides sit on.
    """
    z0, z1 = L["BEAM_Z0"], L["BEAM_Z0"] + L["CROSS_H"]
    ya, yb = L["FRAME_Y"][0][1], L["FRAME_Y"][1][0]
    for i, (x0, x1) in enumerate(L["LEG_X"]):
        _box(W, f"cv_cross_{i}", (x0, x1), (ya, yb), (z0, z1),
             rgba=(0.40, 0.42, 0.46, 1.0))


# -------------------------------------------------------- roller bearings
def bearings(spec, W, L):
    """Blocks carrying the four roller ends, between the belt edge and the rail.

    Geometry unchanged from the verified build - 1 mm clear of the belt, the
    outer face tangent to the rail - only the tail end follows the roller that
    the belt extension moved.
    """
    hx = L["BEAR_HX"]
    for tag, (y0, y1) in (("s", (L["RAIL_IN"][0], L["RAIL_IN"][0] + 0.016)),
                          ("n", (L["RAIL_IN"][1] - 0.016, L["RAIL_IN"][1]))):
        for end, xc in (("tail", L["BELT_TAIL"]), ("head", L["BELT_HALF_LEN"])):
            _box(W, f"cv_bearing_{tag}_{end}", (xc - hx, xc + hx), (y0, y1),
                 (L["CONV_Z"] - 0.030, L["CONV_Z"] + 0.030),
                 rgba=(0.34, 0.37, 0.42, 1.0))


# ------------------------------------------------------------ tail take-up
def takeup(spec, W, L):
    """Belt tensioning at the tail: a cross plate and two threaded screws.

    Each screw is three pieces - stud, nut, stud - rather than a stud with a
    nut threaded onto it, because a nut drawn as a solid around a stud is a
    nut occupying the same space as the stud.  The three pieces are collinear
    and butt end to end, which is what the eye reads as "a nut on a screw".
    """
    x0, x1 = L["FRAME_X0"], L["FRAME_X0"] + 0.020
    ya, yb = L["FRAME_Y"][0][1], L["FRAME_Y"][1][0]
    zc = L["CONV_Z"]
    _box(W, "cv_takeup_plate", (x0, x1), (ya, yb), (zc - 0.040, zc + 0.040),
         rgba=C_PLATE)
    block_face = L["BELT_TAIL"] - L["BEAR_HX"]
    for tag, yc in (("s", L["RAIL_IN"][0] + 0.008),
                    ("n", L["RAIL_IN"][1] - 0.008)):
        nut_mid = x1 + 0.013
        _cyl(W, f"cv_takeup_stud_a_{tag}", (x1, yc, zc), "x", 0.007,
             (nut_mid - x1 - 0.007) / 2, (0.86, 0.87, 0.89, 1.0))
        _cyl(W, f"cv_takeup_nut_{tag}", (nut_mid - 0.007, yc, zc), "x", 0.012,
             0.007, C_STEEL)
        _cyl(W, f"cv_takeup_stud_b_{tag}", (nut_mid + 0.007, yc, zc), "x",
             0.007, (block_face - nut_mid - 0.007) / 2, (0.86, 0.87, 0.89, 1.0))


# ----------------------------------------------------------------- guards
def guards(spec, W, L):
    """Side fences over the infeed and the outfeed, with a folded top lip.

    These started as closed sheet tunnels - two walls and a cap plate - and a
    tunnel seen from the END of the line is a solid slab.  That is exactly how
    it rendered: a mustard box standing on the frame with the belt hidden
    behind it.  A fence does the same job (it keeps hands out of the nip) and
    leaves the belt, the guides and the incoming part visible from the
    operator's side, which is how guards are built on a machine that has to be
    watched while it runs.

    The walls stand on the beams' top faces rather than being bolted beside
    them, so every plate rests on a surface instead of hanging off one.
    """
    top, t = L["BEAM_TOP"], L["GUARD_T"]
    h, lip = L["GUARD_H"], L["GUARD_LIP"]
    # the near wall is inset to GUARD_Y0 (0.156) so it misses the work table's
    # top plate, which reaches y = 0.150 at exactly this height
    wall = (L["GUARD_Y0"], L["GUARD_Y0"] + t)
    wall_n = (L["FRAME_Y"][1][1] - t, L["FRAME_Y"][1][1])
    for tag, (gx0, gx1) in L["GUARD_X"]:
        for side, yr in (("s", wall), ("n", wall_n)):
            _box(W, f"cv_guard_{tag}_{side}", (gx0, gx1), yr,
                 (top, top + h), rgba=C_GUARD)
        # folded lip: sits on the wall's top edge and turns outboard, so the
        # guard is a fence with a return rather than a bare sheet edge
        _box(W, f"cv_guard_{tag}_lip_s", (gx0, gx1),
             (wall[0] - lip + t, wall[1]), (top + h, top + h + t),
             rgba=C_GUARD)
        _box(W, f"cv_guard_{tag}_lip_n", (gx0, gx1),
             (wall_n[0], wall_n[1] + lip - t), (top + h, top + h + t),
             rgba=C_GUARD)
        # warning decal on the near fence, facing the operator's side
        _box(W, f"label_guard_{tag}", (gx0 + 0.020, gx1 - 0.020),
             (wall[0] - 0.0012, wall[0]), (top + 0.021, top + 0.069),
             rgba=(1.0, 1.0, 1.0, 1.0))


def flares(spec, W, L):
    """Flared infeed guides - the funnel that centres a part before the pick.

    Kept verbatim from the verified build: the plate's inner corner lands on
    the rail's inner face (that is what carries it) and it starts 7 mm above
    the belt, which keeps it clear of the tail bearing blocks.  These are the
    only geoms in this file that touch a moving part, so their numbers are not
    a style choice.
    """
    BELT_Y, CONV_X, BELT_TOP = L["BELT_Y"], L["CONV_X"], L["BELT_TOP"]
    lane_in = L["RAIL_Y"] - L["RAIL_HALF_T"] - 0.003
    for tag, s in (("n", -1.0), ("f", 1.0)):
        _plate(W, f"cv_flare_{tag}",
               (CONV_X - 0.72, BELT_Y + s * lane_in),
               (CONV_X - 0.40, BELT_Y + s * 0.038),
               (BELT_TOP + 0.007, BELT_TOP + 0.057), rgba=C_GUARD)


def _plate(W, name, p0, p1, zr, rgba, thick=0.006):
    """A flat plate between two points in XY (the flared side guide)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = float(np.hypot(dx, dy))
    ang = float(np.arctan2(dy, dx))
    g = _box(W, name, (-(length + thick) / 2, (length + thick) / 2),
             (-thick / 2, thick / 2), zr, rgba=rgba)
    g.pos = col([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2,
                 (zr[0] + zr[1]) / 2])
    g.quat = col([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)])
    return g
