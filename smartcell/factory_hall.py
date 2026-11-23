"""Factory hall: floor, walls, columns, beams, purlins, roof, windows, lights.

No open-licensed MJCF factory building could be found, so the hall is built
here from boxes.  Everything is stacked the way a real beam-and-purlin shed is
and - just as important - the shell is CLOSED:

    floor  -> walls (to the eaves line) + steel columns on the long walls
           -> primary beams spanning the long direction, ends butting into
              the end walls
           -> purlins crossing wall to wall, resting on the beams
           -> roof deck on the purlins
           -> lamps hung from the deck on rods

Every part rests on the part below it - verify_cell.py checks exactly that
with mj_geomDistance.  Walls, dado band and window band are SEGMENTED at the
columns: a full-length band would run straight through them (and the verifier
reports it as an interpenetration).  Segmentation also gives the facade its
bays, which is what stops a big wall reading as a grey plane.
"""
import mujoco
import numpy as np

from mjbox import box as _box, col  # noqa: F401  (shared geometry helpers)

# hall envelope, metres
HX = 3.10                    # half length in x  (6.2 m)
HY = 2.60                    # half depth in y   (5.2 m)
WALL_T = 0.15                # wall thickness
WALL_H = 3.20                # eaves height: top of the cladding
BEAM_H = 0.35                # primary beam depth
PUR_H = 0.12                 # purlin depth
DECK_T = 0.06                # roof deck (liner sheet) thickness
EYE = WALL_H + BEAM_H        # top of the beams = underside of the purlins
EAVE = EYE + PUR_H           # underside of the deck = top of the walls
CEIL = EAVE + DECK_T         # top of the hall (= old EYE + ROOF_T)
BEAM_Y = (-1.70, 0.0, 1.70)  # primary beams span x, ends into the end walls
PUR_X = (-2.80, -2.18, -1.30, -0.44, 0.44, 1.30, 2.18, 2.80)  # purlin stations
PUR_HW = 0.04                # purlin half width
COL_X = (-2.55, 0.0, 2.55)   # steel column stations on the long walls
COL_HW = 0.07                # column half width / half depth
WIN_Z = (2.10, 2.85)         # strip window band
DOOR_Y = 0.0                  # roller door centre on the +x wall
DOOR_W, DOOR_H = 2.40, 2.90
BEAM_X = HX - WALL_T         # beams span -BEAM_X..BEAM_X, into the end walls

C_FLOOR = (0.42, 0.42, 0.44, 1.0)   # historical: unused since scene_look
C_MARK = (0.85, 0.68, 0.05, 1.0)
C_WALL = (0.62, 0.63, 0.65, 1.0)
C_WALL_LO = (0.38, 0.40, 0.43, 1.0)
C_BAND = (0.20, 0.26, 0.34, 1.0)
C_STEEL = (0.32, 0.34, 0.37, 1.0)
C_GLASS = (0.72, 0.82, 0.90, 1.0)
C_DOOR = (0.24, 0.34, 0.46, 1.0)
C_LAMP = (0.95, 0.96, 0.98, 1.0)


def _band(W, name, along, at, t, zr, rgba, span=None):
    """Thin panel standing on a wall's inner face (never inside the wall).

    `along` is the axis the panel runs along.  Panels that run along y stop
    `t` short of the corners so they butt the panels that run along x.
    `span` overrides the run length - used to break bands at the columns.
    """
    inner = (HX if along == "x" else HY) - WALL_T - (0.012 if along == "y" else 0.0)
    rng = span if span is not None else (-inner, inner)
    if along == "x":
        _box(W, name, rng, (at, at + t), zr, rgba=rgba)
    else:
        _box(W, name, (at, at + t), rng, zr, rgba=rgba)


def bays():
    """Clear x ranges between the steel columns.

    Columns stand proud of the long walls' inner face, so every full-length
    band (dado, windows) has to stop short of them or it runs through them.
    """
    edges = [-HX + WALL_T]
    for xc in COL_X:
        edges += [xc - COL_HW - 0.02, xc + COL_HW + 0.02]
    edges.append(HX - WALL_T)
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]


def build_hall(spec, cell_box=None):
    """Add the hall to `spec`.  `cell_box` = (x0,x1,y0,y1) floor marking."""
    W = spec.worldbody
    ax = HX - WALL_T

    # floor markings: painted lines 2 mm proud of the slab, they rest on it
    if cell_box:
        x0, x1, y0, y1 = cell_box
        t = 0.06
        _box(W, "mark_s", (x0, x1), (y0 - t, y0), (0.0, 0.002), rgba=C_MARK)
        _box(W, "mark_n", (x0, x1), (y1, y1 + t), (0.0, 0.002), rgba=C_MARK)
        _box(W, "mark_w", (x0 - t, x0), (y0, y1), (0.0, 0.002), rgba=C_MARK)
        _box(W, "mark_e", (x1, x1 + t), (y0, y1), (0.0, 0.002), rgba=C_MARK)
    # pedestrian aisle along the front
    _box(W, "aisle", (-HX + 0.2, HX - 0.2), (-HY + 0.35, -HY + 0.41),
         (0.0, 0.002), rgba=C_MARK)

    # walls: two long (x) and two short (y), standing on the floor and
    # running right up to the roof deck, so the shell has no open eaves band
    _box(W, "wall_s", (-HX, HX), (-HY, -HY + WALL_T), (0.0, EAVE), rgba=C_WALL)
    _box(W, "wall_n", (-HX, HX), (HY - WALL_T, HY), (0.0, EAVE), rgba=C_WALL)
    _box(W, "wall_w", (-HX, -HX + WALL_T), (-HY + WALL_T, HY - WALL_T),
         (0.0, EAVE), rgba=C_WALL)
    _box(W, "wall_e", (ax, HX), (-HY + WALL_T, HY - WALL_T), (0.0, EAVE),
         rgba=C_WALL)
    # steel columns on the long walls, proud of the inner face, floor to deck.
    # They divide the wall into bays and carry the roof line.
    for i, xc in enumerate(COL_X):
        for tag, yin in (("s", -HY + WALL_T), ("n", HY - WALL_T - 2 * COL_HW)):
            _box(W, f"col_{i}_{tag}", (xc - COL_HW, xc + COL_HW),
                 (yin, yin + 2 * COL_HW), (0.0, EAVE), rgba=C_STEEL)
    # dado band and skirting on the inner faces, broken at the columns
    for i, (x0, x1) in enumerate(bays()):
        _band(W, f"dado_s_{i}", "x", -HY + WALL_T, 0.012, (0.0, 1.05),
              C_WALL_LO, span=(x0, x1))
        _band(W, f"dado_n_{i}", "x", HY - WALL_T - 0.012, 0.012, (0.0, 1.05),
              C_WALL_LO, span=(x0, x1))
    _band(W, "dado_w", "y", -HX + WALL_T, 0.012, (0.0, 1.05), C_WALL_LO)
    # strip windows high in the long walls, one pane per bay (glazing proud
    # of the inner face, a real facade reads as windows, not as a light band)
    for i, (x0, x1) in enumerate(bays()):
        _band(W, f"win_s_{i}", "x", -HY + WALL_T, 0.02, WIN_Z, C_GLASS,
              span=(x0, x1))
        _band(W, f"win_n_{i}", "x", HY - WALL_T - 0.02, 0.02, WIN_Z, C_GLASS,
              span=(x0, x1))

    # roller shutter door in the east wall, curtain hung on the inner face.
    # The curtain carries the slat texture (make_textures.shutter); what is
    # built here is the hardware that goes with it - the guides, the drum
    # housing at the top, the bottom bar the curtain lands on, its rubber
    # floor seal, and the pull bar.
    dy0, dy1 = DOOR_Y - DOOR_W / 2, DOOR_Y + DOOR_W / 2
    _box(W, "door_leaf", (ax - 0.03, ax), (dy0, dy1), (0.0, DOOR_H),
         rgba=C_DOOR)
    for s, (gy0, gy1) in (("a", (dy0 - 0.05, dy0)),
                          ("b", (dy1, dy1 + 0.05))):
        _box(W, f"door_guide_{s}", (ax - 0.06, ax), (gy0, gy1),
             (0.0, DOOR_H + 0.12), rgba=C_STEEL)
    _box(W, "door_lintel", (ax - 0.06, ax), (dy0 - 0.08, dy1 + 0.03),
         (DOOR_H + 0.12, DOOR_H + 0.30), rgba=C_STEEL)
    # bottom bar, with its rubber seal standing on the slab, and the pull bar
    _box(W, "door_rail", (ax - 0.08, ax - 0.03), (dy0, dy1), (0.012, 0.088),
         rgba=C_STEEL)
    _box(W, "door_seal", (ax - 0.08, ax - 0.03), (dy0, dy1), (0.0, 0.012),
         rgba=(0.12, 0.12, 0.13, 1.0))
    _box(W, "door_handle_0", (ax - 0.095, ax - 0.080),
         (DOOR_Y - 0.25, DOOR_Y + 0.25), (0.045, 0.065), rgba=C_STEEL)

    # primary beams span the long direction, their ends butting into the end
    # walls' inner faces (they must not run inside them: the walls now reach
    # the deck, and an overlap there is an interpenetration)
    for i, y in enumerate(BEAM_Y):
        _box(W, f"beam_{i}", (-BEAM_X, BEAM_X), (y - 0.09, y + 0.09),
             (WALL_H, EYE), rgba=C_STEEL)
    # purlins cross wall to wall and rest on the beams; the deck sits on them
    for i, xc in enumerate(PUR_X):
        _box(W, f"purlin_{i}", (xc - PUR_HW, xc + PUR_HW),
             (-HY + WALL_T, HY - WALL_T), (EYE, EAVE), rgba=C_STEEL)
    _box(W, "roof", (-HX, HX), (-HY, HY), (EAVE, CEIL), rgba=C_WALL_LO)

    # light fixtures hung from the deck on rods, between the purlins, with
    # real MuJoCo lights.  All four are high-bays aimed down: an UPLIGHT aimed
    # at the deck was built and measured to contribute literally nothing in
    # this renderer (0.0 luminance with and without it, while the same light
    # pointing down changed the picture by 80 counts), so no light slot is
    # spent on one.  The deck is instead lit by the rig and by its own light
    # material - see scene_look.m_roof.
    n = 0
    for y in (-0.85, 0.85):
        for x in (-1.60, 1.60):
            zc = CEIL - 0.55
            _box(W, f"lamp_rod_{n}", (x - 0.012, x + 0.012),
                 (y - 0.012, y + 0.012), (zc + 0.06, EAVE), rgba=C_STEEL)
            _box(W, f"lamp_{n}", (x - 0.30, x + 0.30), (y - 0.10, y + 0.10),
                 (zc, zc + 0.06), rgba=C_STEEL)
            _box(W, f"lamp_lens_{n}", (x - 0.28, x + 0.28),
                 (y - 0.08, y + 0.08), (zc - 0.012, zc), rgba=C_LAMP)
            lt = W.add_light()
            lt.name = f"hall_{n}"
            lt.type = mujoco.mjtLightType.mjLIGHT_SPOT
            lt.pos = col([x, y, zc - 0.02])
            lt.dir = col([0.0, 0.0, -1.0])
            lt.diffuse = col([0.55, 0.56, 0.60])
            lt.specular = col([0.1, 0.1, 0.1])
            lt.castshadow = False
            lt.cutoff = 55.0
            lt.exponent = 8.0
            lt.range = 6.0
            lt.intensity = 22.0
            n += 1

    # NOTE: no ambient fill light here any more.  The old `hall_fill`
    # directional light compiled to intensity 0 (MjSpec/XML default trap) and
    # never lit anything; scene_look.relight() now builds the ambient dome.

    return {"hx": HX, "hy": HY, "wall_t": WALL_T, "wall_h": WALL_H,
            "ceiling": CEIL, "ceil": CEIL, "eave": EAVE, "eye": EYE,
            "door_y": DOOR_Y}


def add_props(spec, hall):
    """A loaded pallet in the far corner: context, and it rests on the floor."""
    W = spec.worldbody
    x0 = -hall["hx"] + 0.75
    y0 = hall["hy"] - 1.05
    pw, pd = 1.20, 0.80
    for i in range(4):                       # deck boards on three bearers
        _box(W, f"pallet_slat_{i}", (x0, x0 + pw),
             (y0 + 0.10 * i, y0 + 0.10 * i + 0.10), (0.10, 0.12),
             rgba=(0.58, 0.47, 0.32, 1.0))
    for i, bx in enumerate((x0, 0.5 * (x0 + x0 + pw) - 0.06, x0 + pw - 0.12)):
        _box(W, f"pallet_block_{i}", (bx, bx + 0.12), (y0, y0 + pd),
             (0.0, 0.10), rgba=(0.48, 0.38, 0.26, 1.0))
    deck = 0.12                              # top of the pallet
    cols = ((0.72, 0.52, 0.30, 1.0), (0.30, 0.48, 0.62, 1.0),
            (0.55, 0.58, 0.35, 1.0))
    # two crates side by side on the deck, one stacked on the first
    for i, (dx, dy) in enumerate(((0.02, 0.06), (0.60, 0.06))):
        _box(W, f"crate_{i}", (x0 + dx, x0 + dx + 0.56),
             (y0 + dy, y0 + dy + 0.68), (deck, deck + 0.30), rgba=cols[i])
    _box(W, "crate_2", (x0 + 0.02, x0 + 0.58), (y0 + 0.06, y0 + 0.74),
         (deck + 0.30, deck + 0.60), rgba=cols[2])
