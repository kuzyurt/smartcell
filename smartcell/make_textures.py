"""Procedural PNG textures for the cell, generated with PIL only.

Every texture is made from deterministic value noise (seeded, so a rebuild is
byte-identical): fBm in 2-3 octaves plus a few edge treatments.  No stock
photos, no external downloads - the provenance chain stays closed.

Textures (metre-based sizes, square, power of two):
  concrete_1080   worn industrial concrete floor, joint grid, scuffs
  wood_1024       pine planks for the table top and the pallets
  steel_1024      brushed steel for rails, stand, lamp bodies
  galv_1024       galvanised sheet metal for guards and bin walls
  rubber_512      dark rubber belt surface with a subtle transverse grain
  cardboard_512   kraft cardboard for the shipping boxes
  plastic_512     blue HDPE for the tote and the discharge bin
  wall_1024       light painted hall wall with panel joints
  shutter_1024    roller-shutter slats for the hall's big door
  alu_1024        brushed aluminium for the line frame
  deck_1024       light metal roof decking / liner panel
  cast_alu_1024   sand-cast aluminium for gearbox and motor housings
  enamel_1024     machine enamel, orange peel and chips
  chequer_1024    raised chequer plate, darkened, for the feet's base pads
  hazard_512      diagonal warning stripe for labels and guard edges

RESOLUTION CAP: no texture may exceed 1920x1080 ("max 1080p").  _check_res()
enforces it on every write, so a generator default cannot quietly break it.
Only the concrete floor - the one surface that fills the frame - uses the
full budget; everything else is 1024 or smaller.

TILE SIZES: no material here carries its own texrepeat.  After paint(),
scene_look._fit_textures() clones a material per geom whose texrepeat lays a
SQUARE tile - the size in `scene_look._TILES`, in metres per material - over
that geom's biggest face, with texuniform = false so the repeat counts
fractions of the face rather than of a metre.  (texuniform = true maps s along
local X and t along local Y, which leaves a 150 mm wall or a roof slab with no
vertical tiling at all - that was the original stretched-texture bug.)
Measured on the compiled model: 422 textured geoms, worst tile aspect 1.000,
so no geom stretches a texture whatever its proportions.  With those tile
sizes a 1024 px texture lands at about 0.5 mm per pixel on the line frame
(0.5 m tile), 1.6 mm per pixel on the floor (3 m tile at 1920 px) and 2.0 mm
per pixel on the walls (2 m tile).

LEVELS matter more than anything else here: a textured material's rgba is
ignored by the renderer, so these generators' absolute brightness IS the
scene's value ladder.  Three of them have been retuned against a measured
render, each documented at the generator: concrete() (0.78 -> 0.41, the floor
had been the brightest surface in the frame), alu() (0.90 -> 0.70, the machine
read as white plastic at 146 where the hall wall measures 142) and chequer()
(0.66 -> 0.36, base pads at 173 were the brightest thing in the frame).  A new
texture's default level is therefore a look decision, not a detail.

Run:  python smartcell/make_textures.py      (writes into smartcell/assets/tex/, ~4 s)
"""
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "tex")

# --------------------------------------------------------------------- noise
_RNG = np.random.default_rng(20260927)


def _value_noise(n, freq, rng):
    """Smooth value noise on an n x n grid at the given lattice frequency."""
    k = max(2, int(freq))
    grid = rng.random((k, k))
    # bilinear upsample with smoothstep for tileable-ish soft noise
    xs = np.linspace(0, k - 1, n, endpoint=False)
    ix = np.floor(xs).astype(int)
    fx = xs - ix
    fx = fx * fx * (3 - 2 * fx)                     # smoothstep
    rows = grid[:, ix] * (1 - fx) + grid[:, ix + 1] * fx     # (k, n)
    ys = np.linspace(0, k - 1, n, endpoint=False)
    iy = np.floor(ys).astype(int)
    fy = ys - iy
    fy = fy * fy * (3 - 2 * fy)
    out = rows[iy, :] * (1 - fy[:, None]) + rows[iy + 1, :] * fy[:, None]
    return out


def _value_noise_xy(n, kx, ky, rng):
    """Value noise with a rectangular lattice: kx small / ky large gives
    features stretched along x, i.e. horizontal streaks (brushed metal)."""
    kx, ky = max(2, int(kx)), max(2, int(ky))
    grid = rng.random((ky, kx))
    xs = np.linspace(0, kx - 1, n, endpoint=False)
    ix = np.floor(xs).astype(int)
    fx = xs - ix
    fx = fx * fx * (3 - 2 * fx)
    rows = grid[:, ix] * (1 - fx) + grid[:, ix + 1] * fx
    ys = np.linspace(0, ky - 1, n, endpoint=False)
    iy = np.floor(ys).astype(int)
    fy = ys - iy
    fy = fy * fy * (3 - 2 * fy)
    return rows[iy, :] * (1 - fy[:, None]) + rows[iy + 1, :] * fy[:, None]


def fbm(n, rng, octaves=((6, 0.5), (12, 0.3), (24, 0.2)), seed=None):
    """Fractal Brownian motion in [0, 1], n x n."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    total = np.zeros((n, n))
    amp = 0.0
    for freq, w in octaves:
        total += w * _value_noise(n, freq, rng)
        amp += w
    total /= amp
    return total


def _to_img(a, lo=None, hi=None):
    a = np.asarray(a, dtype=float)
    if lo is None:
        lo = a.min()
    if hi is None:
        hi = a.max()
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _flat(a):
    """Clip to 0..1 WITHOUT normalising.

    Every generator below writes its field on an ABSOLUTE scale - brushed
    aluminium is meant to sit at 0.90 with +/-0.03 of variation, a casting
    at 0.62.  Running those through _to_img() stretched each one across the
    full 0..1 range, because _to_img divides by (max - min).  Measured on the
    rendered cell: the gearmotor came out as high-contrast black-and-white
    camouflage and the conveyor legs as diagonal candy stripes, from fields
    whose real spread is under 7% of the level.  A texture is multiplied by
    its material colour, so the absolute level is the thing that matters and
    the spread is what should stay small.
    """
    return np.clip(np.asarray(a, dtype=float), 0.0, 1.0)


def _save(arr, name, size=None):
    """Write a greyscale field as a PNG.  `size` forces a resolution, which
    is how every texture is kept at or below 1080p (1920x1080)."""
    a = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    img.save(os.path.join(OUT, name))
    print(f"  {name}  {img.size[0]}x{img.size[1]}")
    return img.size


# Hard ceiling on texture resolution.  The brief is "max 1080p, not more":
# no texture may exceed 1920x1080, which is 2.07 MP.  Everything below is
# checked against this before it is written, so the limit cannot be broken
# by a later edit to a generator default.
MAX_TEX = (1920, 1080)


def _check_res(size):
    if size[0] > MAX_TEX[0] or size[1] > MAX_TEX[1]:
        raise ValueError(f"texture {size} exceeds the 1080p cap {MAX_TEX}")
    return size


def _grey(n, seed, octaves):
    return fbm(n, None, octaves, seed=seed)


# ------------------------------------------------------------------ concrete
def concrete(n=1920):
    """Worn industrial concrete: mid-grey matrix, fine aggregate, joints.

    Generated square then fitted to 1920x1080 - the 1080p cap.  The floor is
    the one surface that fills the frame, so it gets the whole budget; every
    other texture is 1024 or smaller.

    LEVEL is load-bearing, and measured.  For a material that carries an RGB
    texture, this texture IS the surface colour: the renderer ignores the
    material's `rgba` entirely (measured - the floor rendered 231.2/255 with
    `m_floor` at rgba 0.95 and 231.2 with it at 0.47, while scaling THIS
    texture by 0.5 moved the same pixels to 117.6).  So a value is tuned here,
    not in `scene_look`.  The old level (mean 0.78, "pale matrix") made the
    floor render at 231 - the brightest surface in the frame, brighter than
    the machine at 152 and the wall at 142, which is what flattened the whole
    picture.  Real worn concrete is about 0.40-0.45; at a mean of ~0.41 the
    floor renders ~122 and the ladder comes out machine > wall > floor.
    """
    rng = np.random.default_rng(101)
    base = 0.30 + 0.22 * fbm(n, rng, ((4, 0.35), (10, 0.35), (28, 0.2), (64, 0.1)))
    grit = rng.random((n, n))
    speck = (grit > 0.9985).astype(float) * 0.25          # bright aggregate
    pits = (grit < 0.0015).astype(float) * 0.18           # dark pits
    a = base + speck - pits
    # expansion joint at the tile border only, so the texture tiles into
    # one clean groove grid when repeated over the floor
    b = 3
    yy, xx = np.mgrid[0:n, 0:n]
    groove = (xx < b) | (xx > n - 1 - b) | (yy < b) | (yy > n - 1 - b)
    a = np.where(groove, a * 0.72, a)
    # faint tire scuffs along x
    scuff = fbm(n, np.random.default_rng(7), ((3, 0.5), (7, 0.5)))
    band = np.abs(((yy / n) * 6.0) % 1.0 - 0.5)
    a -= 0.05 * scuff * (band > 0.42)
    _save(_flat(a), "concrete_1080.png", size=MAX_TEX)


# ---------------------------------------------------------------------- wood
def wood(n=1024):
    """Pine planks: long grain, plank seams, subtle colour variation."""
    rng = np.random.default_rng(202)
    yy, xx = np.mgrid[0:n, 0:n]
    plank_h = n // 4
    # per-plank tone offset
    tones = 0.86 + 0.20 * rng.random(4)
    tone = tones[(yy // plank_h) % 4]
    # Grain lines run ALONG the plank, i.e. along x, so the ring index has to
    # depend on y and only wander slowly.  The old version indexed the rings
    # on x with a phase that varied by 6*grain (up to 9 radians), so the
    # "grain" was scrambled from one cycle to the next and the planks tiled
    # as orange camouflage - visible on every crate in the cell.
    warp = _value_noise(n, 3, np.random.default_rng(305))       # slow wander
    fine = _value_noise_xy(n, 90, 5, np.random.default_rng(306))  # along x
    ring = yy / n * 44.0 + 1.6 * warp
    streak = np.abs(np.sin(ring * np.pi)) ** 0.7
    a = 0.62 * tone + 0.24 * (0.40 + 0.60 * streak) * tone + 0.14 * fine
    # seams between planks
    seam = (yy % plank_h < 2) | (yy % plank_h > plank_h - 3)
    a = np.where(seam, a * 0.55, a)
    # occasional knot: dark ellipse
    kx, ky = int(0.31 * n), int(0.37 * n)
    knot = ((xx - kx) / (0.018 * n)) ** 2 + ((yy - ky) / (0.010 * n)) ** 2
    a = np.where(knot < 1.0, a * 0.62, a)
    # map to pine colour
    r = 0.72 * a + 0.10
    g = 0.58 * a + 0.07
    b = 0.38 * a + 0.05
    _save(_flat(np.dstack([r, g, b])), "wood_1024.png")


# --------------------------------------------------------------------- steel
def steel(n=1024):
    """Brushed steel: horizontal streaks, mild tonal bands."""
    rng = np.random.default_rng(404)
    streaks = _value_noise(n, 3, rng)
    fine = _value_noise(n, 48, np.random.default_rng(405))
    yy, xx = np.mgrid[0:n, 0:n]
    brush = 0.5 + 0.5 * np.sin(xx / n * 260.0 * (0.8 + 0.4 * streaks))
    a = 0.55 + 0.18 * brush + 0.12 * fine
    a += 0.05 * (fbm(n, rng, ((3, 0.6), (8, 0.4)), seed=406) - 0.5)
    _save(_flat(a), "steel_1024.png")


# ---------------------------------------------------------------- galvanised
def galv(n=1024):
    """Galvanised sheet: spangle crystals over a mid grey."""
    rng = np.random.default_rng(505)
    spangle = fbm(n, rng, ((8, 0.6), (20, 0.4)), seed=506)
    spangle = (spangle > 0.52).astype(float)
    # cell borders darker: edge detect by gradient magnitude
    gy, gx = np.gradient(spangle)
    edge = np.clip(np.hypot(gx, gy) * 8.0, 0, 1)
    base = 0.62 + 0.16 * (fbm(n, rng, ((5, 0.5), (14, 0.5)), seed=507) - 0.5)
    a = base - 0.10 * edge + 0.06 * spangle
    _save(_flat(a), "galv_1024.png")


# -------------------------------------------------------------------- rubber
def rubber(n=512):
    """Belt rubber: near-black with a faint transverse grain."""
    rng = np.random.default_rng(606)
    base = 0.16 + 0.06 * fbm(n, rng, ((6, 0.5), (18, 0.5)), seed=607)
    yy, xx = np.mgrid[0:n, 0:n]
    trans = 0.03 * np.sin(yy / n * 90.0) * (0.6 + 0.4 * fbm(n, rng, ((9, 1.0),), seed=608))
    a = base + trans
    _save(_flat(a), "rubber_512.png")


# ----------------------------------------------------------------- cardboard
def cardboard(n=512):
    """Kraft cardboard: warm fibre noise, subtle fluting."""
    rng = np.random.default_rng(707)
    fibre = fbm(n, rng, ((7, 0.5), (17, 0.3), (34, 0.2)), seed=708)
    yy, xx = np.mgrid[0:n, 0:n]
    flute = 0.04 * np.sin(xx / n * 55.0 + 3.0 * fibre)
    a = 0.62 + 0.20 * (fibre - 0.5) + flute
    r = np.clip(a * 1.00 + 0.06, 0, 1)
    g = np.clip(a * 0.78 + 0.03, 0, 1)
    b = np.clip(a * 0.55, 0, 1)
    _save(np.dstack([r, g, b]), "cardboard_512.png")


# ------------------------------------------------------------------- plastic
def plastic(n=512):
    """Blue HDPE: very smooth, faint mould swirl."""
    rng = np.random.default_rng(808)
    a = 0.60 + 0.05 * fbm(n, rng, ((5, 0.6), (13, 0.4)), seed=809)
    r = np.clip(a * 0.30, 0, 1)
    g = np.clip(a * 0.62, 0, 1)
    b = np.clip(a * 0.95, 0, 1)
    _save(np.dstack([r, g, b]), "plastic_512.png")


# ---------------------------------------------------------------------- wall
def wall(n=1024):
    """Light painted hall wall with a panel joint grid.

    Deliberately LOW contrast: the wall is a huge surface seen from far away,
    and any noise that is visible at that distance reads as dirt.  The only
    strong feature is the joint line along the tile's left and top edge, which
    tiles into a regular grid of ~2 m cladding panels - that grid is what
    makes the wall read as a built wall instead of a grey plane.
    """
    rng = np.random.default_rng(909)
    a = 0.94 + 0.035 * fbm(n, rng, ((5, 0.45), (13, 0.35), (34, 0.2)),
                           seed=910)
    yy, xx = np.mgrid[0:n, 0:n]
    # the faint drag of a paint roller: long, almost invisible waves
    a -= 0.010 * (np.sin(yy / n * 5.0 * np.pi) * 0.5 + 0.5)
    # panel joint on the tile's left and top edge (tiles into a grid)
    j = 3
    vert = (xx < j) | (xx > n - 1 - j)
    horiz = (yy < j) | (yy > n - 1 - j)
    a = np.where(vert | horiz, a * 0.80, a)          # recessed joint
    lip_v = (xx >= j) & (xx < j + 3)
    lip_h = (yy >= j) & (yy < j + 3)
    a = np.where(lip_v | lip_h, a * 1.03, a)         # catch light on the lip
    _save(_flat(a), "wall_1024.png")


# -------------------------------------------------------------- aluminium
def alu(n=1024):
    """Brushed aluminium for the assembly line's frame.

    Anisotropic streaks: long along x, fine across y.  Every frame face on the
    line runs its length along the texture's x, so the brush marks follow the
    extrusion, which is what makes it read as anodised profile.

    LEVEL is load-bearing, exactly as it is for concrete(): a textured material
    ignores its rgba, so this generator's level IS the metal's diffuse value.
    It used to sit at 0.90, which measured 146/255 rendered - brighter than the
    hall wall (142) and the floor (123), i.e. an extruded frame that read as
    WHITE PLASTIC rather than metal.  Mill-finish 6063 extrusion is about
    0.60-0.70 diffuse.  At 0.70 the frame measures 116/255 and the line reads
    as a mid-toned machine against a light hall, which is what the reference
    photo of a real line looks like.

    The VARIATION is the other half of it, and it is anisotropic on purpose:
    four layers at a ~4:1 frequency ratio, all long in x, for a spread of
    sd 8/255 (p5-p95 = 27 counts).  The first version measured sd 2.4/255 - a
    near-flat fill - and rendered as untextured grey sheet at every distance,
    which is exactly how the machine came to read as one slab.  Streaks are
    safe where isotropic blotches were not: they average back to the level
    under minification, whereas the cast-aluminium texture's blotches aliased
    into camouflage.
    """
    _save(_brush(n, 0.70, 1111), "alu_1024.png")
    # the same brush, darker: leg tubes, tie bars and the cable duct.  A 50 mm
    # tube shows about a tenth of a tile, so a coarse pattern magnifies into
    # corrugation there - which is what the steel sheet did on the legs - while
    # a fine brush just reads as grain.
    _save(_brush(512, 0.52, 1113), "alu_dk_512.png")


def _brush(n, level, seed):
    """Anisotropic brushed-metal field at an ABSOLUTE level, streaks along x."""
    rng = np.random.default_rng(seed)
    s1 = _value_noise_xy(n, 3, 90, rng)              # the visible brush
    s2 = _value_noise_xy(n, 7, 340, rng)             # mid streaks
    s3 = _value_noise_xy(n, 2, 1500, rng)            # sub-pixel grain
    band = _value_noise_xy(n, 5, 3, rng)             # slow drift down the sheet
    a = (level + 0.115 * (s1 - 0.5) + 0.075 * (s2 - 0.5)
         + 0.030 * (s3 - 0.5) + 0.045 * (band - 0.5))
    # a few dead-straight scratches, indexed modulo n so the tile still tiles
    for _ in range(16):
        ys = (np.arange(1 if rng.random() < 0.75 else 2)
              + int(rng.integers(0, n))) % n
        xs = (np.arange(int(rng.integers(n // 5, n)))
              + int(rng.integers(0, n))) % n
        a[np.ix_(ys, xs)] -= 0.075 if rng.random() < 0.8 else -0.055
    return _flat(a)


# ------------------------------------------------------------------ roof deck
def deck(n=1024):
    """Light metal roof decking, seen from below as the hall ceiling.

    A shallow trapezoidal profile every tile-width/12, plus the same joint
    grid as the walls.  Low contrast on purpose: real roof decks are a pale
    grey and read as ribs, not as stripes.
    """
    rng = np.random.default_rng(1212)
    yy, xx = np.mgrid[0:n, 0:n]
    a = 0.84 + 0.035 * fbm(n, rng, ((6, 0.5), (18, 0.5)), seed=1213)
    t = (xx / n * 12.0) % 1.0                        # 12 ribs per tile
    ribs = np.where(t < 0.35, 0.045,               # crest
                    np.where(t < 0.5, -0.055,      # near flank in shade
                             np.where(t < 0.85, 0.0, -0.075)))   # valley
    a += ribs
    a -= 0.010 * (np.sin(yy / n * 3.0 * np.pi) * 0.5 + 0.5)
    j = 3
    a = np.where((xx < j) | (yy < j), a * 0.86, a)
    _save(_flat(a), "deck_1024.png")


# --------------------------------------------------- cast aluminium (housings)
def cast_alu(n=1024):
    """Sand-cast aluminium: the gearbox and motor housings.

    Castings are not brushed - they have a pebbly skin from the mould, with
    occasional flat machined pads where a face was skimmed.  That contrast
    between pebbled and flat is what makes a casting read as a casting
    rather than as a painted box.
    """
    rng = np.random.default_rng(3131)
    # Pebble frequencies are chosen for the 0.12 m tile this material uses:
    # 12/24/48 per tile is 10/5/2.5 mm of sand-cast skin, which still reads at
    # 1080p up close.  The old 48/96/170 was 0.7 mm at that tile - sub-pixel at
    # any normal camera distance, so it aliased into salt-and-pepper.
    pebble = fbm(n, rng, ((12, 0.45), (24, 0.35), (48, 0.20)), seed=3132)
    # machined pads: a few broad areas skimmed flat and slightly brighter
    pad = fbm(n, np.random.default_rng(3133), ((3, 0.5), (5, 0.5)))
    flat = pad > 0.62
    a = 0.62 + 0.055 * (pebble - 0.5) * 2.0
    a = np.where(flat, 0.655 + 0.030 * (pebble - 0.5), a)
    # sparse casting pits
    grit = rng.random((n, n))
    a -= np.where(grit > 0.9975, 0.13, 0.0)
    _save(_flat(a), "cast_alu_1024.png", size=_check_res((n, n)))


# ------------------------------------------------- machine enamel (paintwork)
def enamel(n=1024):
    """Industrial machine enamel: orange-peel in the film, chips at the edges.

    Orange peel is the low-frequency ripple every sprayed finish has; the
    chips are sparse and expose a duller primer underneath, which is what
    stops a painted panel looking like flat vinyl.
    """
    rng = np.random.default_rng(4141)
    peel = fbm(n, rng, ((9, 0.5), (20, 0.3), (44, 0.2)), seed=4142)
    fine = fbm(n, None, ((60, 0.5), (130, 0.5)), seed=4143)
    a = 0.80 + 0.075 * (peel - 0.5) * 2.0 + 0.030 * (fine - 0.5) * 2.0
    # chips: small blobs where the topcoat has come off
    yy, xx = np.mgrid[0:n, 0:n]
    chipfield = fbm(n, np.random.default_rng(4144), ((14, 0.5), (30, 0.5)))
    chip = (chipfield > 0.70) & (rng.random((n, n)) > 0.55)
    a = np.where(chip, a * 0.62, a)
    _save(_flat(a), "enamel_1024.png", size=_check_res((n, n)))


# ------------------------------------------------------ chequer plate (tread)
def chequer(n=1024):
    """Raised-tear-drop chequer plate for the levelling feet's base pads.

    The real pattern is two crossing sets of short bars at 45 degrees,
    staggered row to row.  Each bar is drawn as a lit top edge and a shadowed
    underside, because at this scale the highlight is what sells it.

    LEVEL: this is the third level retune (after concrete and aluminium), and
    for the same reason - a textured material ignores its rgba, so the level
    here IS the pad's colour.  At 0.66 the base pads measured 173/255, the
    BRIGHTEST surface in the whole frame, ahead of the paint markings; eight
    90 mm pads render the line as a row of white paddles.  Hot-rolled tread
    plate in service is dark, around 0.40.  At 0.36 measured ~... see
    assets/PROVENANCE.md for the number after the rebuild.
    """
    yy, xx = np.mgrid[0:n, 0:n]
    u = (xx + yy) / n
    v = (xx - yy) / n
    a = 0.36 + 0.05 * fbm(n, None, ((30, 0.5), (70, 0.5)), seed=5151)
    bars = np.zeros((n, n))
    # set A: short bars stepping along u, 8 per tile
    for k in range(8):
        c = (k + 0.5) / 8.0
        d = np.abs(((u - c + 0.5) % 1.0) - 0.5)
        m = (d < 0.055) & (np.abs(((v * 5.0) % 1.0) - 0.5) < 0.20)
        bars = np.maximum(bars, m.astype(float) * (1.0 if k % 2 == 0 else 0.85))
    # set B: the other diagonal, offset half a row
    for k in range(8):
        c = (k + 0.0) / 8.0
        d = np.abs(((v - c + 0.5) % 1.0) - 0.5)
        m = (d < 0.050) & (np.abs(((u * 5.0 + 0.5) % 1.0) - 0.5) < 0.17)
        bars = np.maximum(bars, m.astype(float) * 0.9)
    a = a + 0.16 * bars - 0.07 * np.roll(bars, 3, axis=0)   # lit / shadowed
    _save(_flat(a), "chequer_1024.png", size=_check_res((n, n)))


# --------------------------------------------- roller shutter (the big door)
def shutter(n=1024):
    """Roller-shutter curtain: horizontal slats, crown to under-lip.

    The hall's 2.4 x 2.9 m roller door used to carry the galvanised-sheet
    texture, which is a flat spangle field at any scale - a big door with no
    door in it.  What makes a shutter read as a shutter is the SLAT: a bright
    crown where the roll folds over, a face falling away into shadow, and a
    dark line where one slat laps under the next.  That is one vertical
    profile, repeated.

    SLAT PITCH is set by the generator (14 slats per tile) together with
    scene_look._TILES["m_door"], which lays the tile 1.0 m over the door's
    2.9 m face - three whole tiles, so the slats stay uniform at 69 mm and the
    texture never cuts mid-slat at the top of the door.

    LEVEL is the same rule as everywhere else - the generator's level is the
    surface colour - but the door is the one surface in the hall the light rig
    cannot reach: it is on the east wall at x = 2.95 and every lamp, sky dome
    and the key light are aimed at the cell, so the door sees them at a
    grazing angle from 3-4 m away.  Measured with the first level (mean 0.46,
    a real shutter's grey) the whole curtain rendered 20-30/255 - a black
    rectangle with slats you have to look for.  At a mean of 0.60 it renders
    as a mid-grey curtain, which is what the shutter looks like from inside a
    shed whose doors face the sun.
    """
    rng = np.random.default_rng(7171)
    yy, xx = np.mgrid[0:n, 0:n]
    t = (yy / n * 14.0) % 1.0            # position within one slat
    prof = np.where(t < 0.12, 1.00,          # the crown, catching the light
             np.where(t < 0.55, 0.82,        # the slat's face
              np.where(t < 0.78, 0.60,       # rolling into shadow
               np.where(t < 0.92, 0.28,      # under-lip, in its own shadow
                        0.14))))             # the lap line between slats
    a = 0.69 * prof + 0.14
    # fine horizontal grain, as rolled steel has
    a += 0.030 * (_value_noise_xy(n, 90, 4, rng) - 0.5)
    # a heavier joint every 7 slats, as a real curtain is assembled in sections
    a = np.where((yy % (n // 2)) < 3, a * 0.74, a)
    # weathering: water and grime drain down the curtain
    a -= 0.045 * (fbm(n, np.random.default_rng(7172), ((5, 0.5), (13, 0.5)))
                  - 0.5)
    _save(_flat(a), "shutter_1024.png", size=_check_res((n, n)))


# --------------------------------------------------- hazard stripe (labels)
def hazard(n=512):
    """Diagonal yellow/black warning stripe for labels and guard edges.

    Small on purpose: it only ever appears on decals, so 512 is plenty and
    keeps the stripe edges crisp.
    """
    yy, xx = np.mgrid[0:n, 0:n]
    band = ((xx + yy) / n * 7.0) % 1.0
    a = np.where(band < 0.5, 0.88, 0.13)
    a += 0.035 * fbm(n, None, ((40, 0.5), (90, 0.5)), seed=6161)
    b = 6
    edge = (xx < b) | (xx > n - 1 - b) | (yy < b) | (yy > n - 1 - b)
    a = np.where(edge, 0.06, a)
    _save(_flat(a), "hazard_512.png", size=_check_res((n, n)))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("textures ->", OUT)
    concrete()
    wood()
    steel()
    galv()
    rubber()
    cardboard()
    plastic()
    wall()
    shutter()
    alu()
    deck()
    cast_alu()
    enamel()
    chequer()
    hazard()
    print("done")


if __name__ == "__main__":
    main()
