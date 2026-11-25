"""Phase 4/7 dataset: N x 96 x 96 gray stacks + 8 x 6 action chunks.

Primary path reads cache/ep_XXXXXX.npz (precomputed gray frames, action
chunks, qpos - see build_cache.py).  Phase 7 raised the observation grid from
10 Hz to 25 Hz: 40 ms is exactly 8 physics steps, which 24 Hz is not, and the
whole chain (camera, cache rows, chunk entries, eval decisions) runs on that
single grid.  The stack is [j, j-1, ...] on the cache grid, newest first,
matching evaluate.py.  Split is EPISODE-level (rule 4) and asserted disjoint.
"""
import json
import os

import numpy as np
import torch

RES = 96
D_RES = 96              # grid_sample coordinate grid for the wrist-view rotation
# manifold width, set by train.py --aug-sigma / --aug-rot-deg
AUG_SIGMA = 0.015       # rad of N(0, s) on IST and SOLL (uniform over rows)
AUG_ROT_DEG = 2.0       # deg of U(-a, a) wrist-view rotation
AUG_GAP_K = 3.0          # DAgger-lite: scale the SOLL-IST gap by U(0, K)
AUG_DROP_P = 0.0        # proprio dropout: zero the 12-dim IST||SOLL vector
                        # with prob p (train only, no rescale).
                        # Copycat remedy (E2 measured 8.7x proprio reliance):
                        # the net must not be able to solve entry 0 by
                        # reading SOLL. 0.0 = off (default recipe).
P_MOVE = 0.55           # fraction of samples drawn from the moving rows
P_TRANS = 0.0           # fraction drawn from grip-transition rows
                        # (close/open edge inside the chunk, E1: 3% of rows,
                        # 7% of loss). 0.0 = off (default recipe).
P_DESC = 0.0            # fraction drawn from pre-release descent rows
                        # (the last DESC_ROWS rows before the release edge:
                        # look-hold + final 60 mm + release settle. The dip
                        # the hover stall never takes lives here - measured
                        # 15/20 place failures on 0033 hover over the tote.
                        # 0.0 = off (default recipe).
DESC_ROWS = 32          # window before the release edge (rows at 25 Hz)
# Audit: AUG_SIGMA_MOVING (0.09) and AUG_SPEED_K (0.6) USED to live
# here and were printed every epoch as "+-0.09 rad on rows that move" while
# NOTHING in the code applied them - every model v3..v7 trained with uniform
# +-aug_sigma.  They are deleted rather than implemented: the feature is
# unproven (it was credited for a model that still scored 0 picks), and a
# constant that only decorates a log line is how this codebase lies to itself.
STACK = 3               # frames per sample -> (STACK-1)*GRID_MS of history
CHUNK = 8
GRID_MS = 40            # observation/policy grid: 25 Hz, 8 physics rows
GRID_S = GRID_MS / 1000.0


def set_stack(n):
    """Phase 6/7: history length in frames (frames are GRID_MS apart).

    At 10 Hz the expert's pre-close park at median
    300 ms; 8 frames of 100 ms spanned it in every episode, which is why the
    Phase-6 model used 8.  At 25 Hz the same 8 frames span 280 ms - the park
    region is no longer fully observable from the stack alone, and it does not
    have to be: the Phase-7 grasp fix moved the burden from timing to a
    +-20 mm geometric envelope (see collect_state.jaw_center_xy).
    """
    global STACK
    assert 2 <= int(n) <= 32, n
    STACK = int(n)


def load_manifest(ds="ds_0001"):
    rows = [json.loads(l) for l in open(os.path.join(ds, "manifest.jsonl"))]
    return [r for r in rows if r.get("img")]


def split_ids(rows):
    """Train / val / test by EPISODE, 80 / 10 / 10, deterministic.

    Never by frame: consecutive frames of one episode overlap through the
    frame stack, so a frame split leaks the target into its own input.

    It used to be `ep_id % 20` -> <17 / 17-18 / >=19.  That is only
    proportional while the ids span 20 or more, and ds_0007 has 14 episodes
    (ids 0..13), so val and test came back EMPTY and BCDataset died on
    "cache holds set()" -- a small dataset silently had no way to select a
    checkpoint.  Rank-based strides give every dataset size the same 80/10/10
    with at least one episode held out, and are disjoint by construction.

    On ds_0006 (453 episodes) this is 387 / 44 / 22 against the old
    385 / 46 / 22 for train/val/test.  No checkpoint was ever trained on
    ds_0006, so nothing already measured is invalidated.
    """
    ids = sorted(int(r["ep_id"]) for r in rows)
    assert len(ids) >= 3,         f"need >=3 episodes to hold out val and test, got {len(ids)}"
    va = set(ids[9::10])                        # 10%
    rest = [i for k, i in enumerate(ids) if k % 10 != 9]
    te = set(rest[8::9])                        # ~10% of what is left
    tr = set(rest) - te
    assert not (tr & va or tr & te or va & te),         "rule 4: episode sets must be disjoint"
    assert tr and va and te,         f"split is empty: train {len(tr)} val {len(va)} test {len(te)}"
    return tr, va, te


_FRAMES = {}


def _frames_for(cache_dir):
    """(ids, offsets, memmap) for the consolidated frame store, or None.

    Shared on purpose: the memmap is opened once per PROCESS, and Windows
    spawns one process per DataLoader worker, so every worker maps the same
    1.18 GB file and the OS page cache serves them all.  Without it each worker
    decompresses its own copy and the loader costs ~7 GB at 6 workers.
    """
    key = os.path.abspath(cache_dir)
    if key not in _FRAMES:
        fp = os.path.join(cache_dir, "frames.npy")
        _FRAMES[key] = None
        if os.path.exists(fp):
            try:
                _FRAMES[key] = (
                    np.load(os.path.join(cache_dir, "frames_ids.npy")),
                    np.load(os.path.join(cache_dir, "frames_off.npy")),
                    np.load(fp, mmap_mode="r"))
            except (OSError, ValueError):
                _FRAMES[key] = None
    return _FRAMES[key]


class CacheEp:
    def __getstate__(self):
        """Never ship frames to a spawned worker.

        This is the freeze.  A worker is a SPAWNED process, so the dataset is
        pickled to it, and pickling a memmap materialises it: one CacheEp came
        out at 1211 MB.  Six workers were therefore 7 GB of pickles on top of
        six private decompressions - the machine paged itself to death.
        Store the offset pair instead and let the child mmap the same file:
        the pages are then shared, and the pickle is a few hundred bytes.
        """
        d = self.__dict__.copy()
        d["_frame_spec"] = (self._fr[0], self._fr[1]) if self._fr else None
        d["_cache_dir"] = self._cache_dir
        d["_fr"] = None
        d["_gz"] = None
        d["g"] = None
        return d

    def __setstate__(self, d):
        spec = d.pop("_frame_spec", None)
        cdir = d.pop("_cache_dir", None)
        self.__dict__.update(d)
        fr = _frames_for(cdir) if cdir else None
        if fr is not None and spec is not None:
            self._fr = (spec[0], spec[1], fr[2])
            self.g = fr[2][spec[0]:spec[1]]
        else:                       # no consolidated store: load the npz
            z = np.load(f"{cdir}/ep_{self.ep_id:06d}.npz")
            self._fr = None
            self._gz = z["g"]
            self.g = self._gz

    def _stack_needed(self):
        # rows below this have no full history; the dataset never samples them
        return int(getattr(self, "_stk", 3)) - 1

    def _split(self):
        """Phase pools + how far the arm is ABOUT TO move (needs _stk, so it
        runs after construction rather than inside __init__).

        The width has to track the UPCOMING motion, not the instantaneous
        speed.  The expert's chase is committed one chunk BEFORE the arm
        moves: at cache row 236 (t=9.44 s) the arm is still exactly at the
        wait pose and |dq| is still zero, but the row's own chunk already
        sweeps 0.727 normalised (1.27 rad) in joint 1.  Keying the width on
        |dq| or on the row-to-row speed gave those transition rows the
        SMALLEST sigma - precisely where the rollout arrives ~0.1 rad off the
        wait pose and needs the widest basin.  Measured: the policy committed
        0.30 rad of the 2 rad sweep and then reverted, 100% of the time.
        """
        dq = np.abs(self.q[:, :5] - self.q[0, :5]).max(axis=1)
        lo = self._stack_needed()
        # rad the arm travels over this row's own 8-entry (320 ms) chunk
        self.motion = np.zeros(self.n)
        if self.n > 1:
            # rad travelled per row, so the 8-row window sums to rad/chunk
            step = np.abs(np.diff(self.q[:, :5], axis=0)).max(axis=1)
            fut = np.convolve(step[::-1], np.ones(8), "valid")[::-1]
            self.motion[:len(fut)] = fut[:self.n]
        moving = (dq > 0.02) | (self.motion > 0.25)
        self.moving = np.array([j for j in np.nonzero(moving)[0] if j >= lo])
        self.still = np.array([j for j in np.nonzero(~moving)[0] if j >= lo])
        # grip-transition rows: the close/open edge falls inside this row's
        # own 8-entry chunk label (grip range > 0.5 normalised). E1: 3.12%
        # of val rows; the trigger the policy must learn lives here.
        try:
            grange = (self.chunks[:, :, 5].max(axis=1)
                      - self.chunks[:, :, 5].min(axis=1))
            self.trans = np.array(
                [j for j in np.nonzero(grange > 0.5)[0] if j >= lo])
        except Exception:
            self.trans = np.array([], dtype=np.int64)
        # pre-release descent rows: the shut stretch's tail (look-hold +
        # final descend + release settle). chunks[j] holds 8 FORWARD grip
        # entries, so rows whose window mean is shut mark jaws-shut; the
        # stretch end + 1 is the release edge. Flat carry, diagonal and
        # dagger legs all end the shut stretch the same way, so one
        # reopen-anchored window covers every expert path while excluding
        # the carry hover (flat: hover ends ~reopen-32 by construction).
        try:
            gmean = self.chunks[:, :, 5].mean(axis=1)
            shut = np.nonzero(gmean < -0.4)[0]
            shut = shut[shut >= lo]
            if len(shut):
                reopen = int(shut[-1]) + 1
                self.desc = np.array(
                    [j for j in range(max(int(lo), reopen - DESC_ROWS),
                                     min(reopen, self.n))])
            else:
                self.desc = np.array([], dtype=np.int64)
        except Exception:
            self.desc = np.array([], dtype=np.int64)

    def __init__(self, ep_id, cache_dir="cache", stack=3):
        self.ep_id = int(ep_id)
        self._cache_dir = cache_dir
        z = np.load(f"{cache_dir}/ep_{ep_id:06d}.npz")
        fr = _frames_for(cache_dir)
        self._fr = None
        if fr is not None:
            ids, off, mm = fr
            w = np.nonzero(ids == int(ep_id))[0]
            if len(w):
                k = int(w[0])
                self._fr = (int(off[k]), int(off[k + 1]), mm)
        # CRITICAL: do NOT touch z["g"] when the shared memmap is present.
        # Reading it decompresses the whole episode - the cache is stored
        # compressed (166 MB on disk) and expands to 1.17 GB - so every
        # worker paid the full image array anyway and the memmap bought
        # nothing.  Measured: 4.11 GB resident before a single epoch ran.
        if self._fr is not None:
            self._gz = None
            self.g = self._fr[2][self._fr[0]:self._fr[1]]
        else:
            self._gz = z["g"]          # (N,96,96) uint8, fallback only
            self.g = self._gz
        self.chunks = z["chunks"]      # (N,8,6) float32
        self.q = z["q"]                # (N,6) float32  IST, radians
        # SOLL.  Absent means the cache predates build_cache.py writing it, and
        # the model's proprio is 12 wide now (IST 6 || SOLL 6), so a silent 6
        # would train a policy that reads a wrong-width vector.  Fail loudly.
        if "soll" not in z:
            raise KeyError(
                f"{cache_dir}/ep_{ep_id:06d}.npz has no 'soll' field; it was "
                f"built before SOLL was carried into the cache.  Rebuild it "
                f"(build_cache.py --ds <ds> --out {cache_dir}) - the model's "
                f"proprio is IST 6 || SOLL 6 = 12 wide.")
        self.soll = z["soll"]          # (N,6) float32  SOLL, SAME units as q
        assert self.soll.shape[1] == self.q.shape[1], \
            f"IST {self.q.shape} and SOLL {self.soll.shape} must match"
        self.n = len(self.g)
        # Phase split.  The expert parks at ONE pose for the first 9.5 s of an
        # 18 s episode, so a uniform row sampler spends 55% of its gradient on
        # "hold still" and ~7% on the 0.9 s chase - the 2 rad/s sweep that
        # actually decides whether the cell works.  Measured on the rollout:
        # with a uniform sampler the policy waits perfectly and then lunges
        # 0.3 rad at t=8.8 s and retreats instead of committing.
        # `moving` / `still` / `speed` are built in _split(), once the stack
        # size is known: the phase pools must exclude rows with no history.
        # the row grid is the one thing the model's history arithmetic assumes
        if self.n > 1:
            dt = float(np.diff(z["t"]).mean())
            assert abs(dt - GRID_MS / 1000.0) < 0.0025, (
                f"{cache_dir}/ep_{ep_id:06d}.npz rows are {dt * 1000:.1f} ms "
                f"apart, the grid is {GRID_MS} ms")
        # dataset tag: episode ids OVERLAP across ds_0001/0002/0003, so a
        # cache dir from another dataset reads perfectly but is the wrong data
        self.ds = str(z["ds"]) if "ds" in z else None


class BCDataset(torch.utils.data.Dataset):
    def __init__(self, ids, augment=True, stack=None, cache_dir="cache",
                 ds=None, allow_untagged=False, aug_sigma=0.015,
                 aug_rot_deg=2.0, aug_gap_k=3.0, aug_drop_p=0.0,
                 trans_p=0.0, desc_p=0.0):
        self.eps = [CacheEp(i, cache_dir) for i in sorted(ids)]
        for e in self.eps:      # moving/still pools need the stack size
            e._stk = int(stack if stack is not None else STACK)
            e._split()
        tag = {e.ds for e in self.eps}
        if ds is not None and tag != {ds}:
            # untagged = built before build_cache.py wrote the ds field; usable
            # only with an explicit opt-in so a mix-up cannot pass silently
            assert tag == {None} and allow_untagged, (
                f"cache {cache_dir} holds {tag}, asked for {ds}"
                + (" (untagged cache: pass allow_untagged=True to accept)"
                   if tag == {None} else ""))
            print(f"WARNING: {cache_dir} is untagged, assuming {ds}")
        self.augment = augment
        # the stack lives on the instance, not only in the module global:
        # Windows DataLoader workers are spawned, so a global set by
        # set_stack() at runtime does NOT reach them and they would fall back
        # to the default (silently feeding 3 frames to an 8-channel model).
        self.stack = int(stack if stack is not None else STACK)
        # MANIFOLD WIDTH.  The demonstrations hold the arm at ONE pose for the
        # whole 8 s wait (measured: 1 unique pose, <=5e-4 rad of jitter over
        # 18183 rows) because the expert commands a constant target and the
        # servo parks exactly on it.  That is a zero-width manifold: the
        # policy's own ~7e-3 rad of output noise moves the rollout off it in
        # one decision, the off-manifold image is OOD, and the map EXPANDS
        # instead of contracting (shown its own 5.6e-3 rad error the model
        # commanded 7e-3 rad further off) - dq grew 0.0002 -> 0.006 -> 0.017
        # -> 0.065 -> 0.19 -> 0.49 rad in five 80 ms decisions.
        # Pinning the arm with a command deadband (evaluate.py, since removed)
        # let it wait the full 9 s and start the chase on time, which is what
        # proved the wait logic is fine and the manifold is the problem.
        # The fix is to train on a NEIGHBOURHOOD: perturb IST/SOLL by
        # aug_sigma rad and rotate the wrist view by aug_rot_deg, which is
        # what a wrist offset actually looks like.  Consistent image +
        # proprio perturbation is the point - perturbing one alone teaches the
        # model to ignore the channel it must trust.
        self.aug_sigma = float(aug_sigma)
        self.aug_rot_deg = float(aug_rot_deg)
        self.aug_drop_p = float(aug_drop_p)
        self.trans_p = float(trans_p)
        self.desc_p = float(desc_p)
        # DAgger-lite gap scaling: in the rollout SOLL is the policy's OWN last
        # command fed back, which moves the SOLL-IST gap into bands the expert
        # never visits (measured: expert gap is 0 on wait rows,
        # 1.3-2.3 rad for the 6 transition rows, then <=0.2; the band 0.3-1.2
        # never occurs and is exactly where the policy's lunge dies).  Scaling
        # the row's OWN gap by U(0, aug_gap_k) covers every band with the row's
        # own image+IST+label, so "chase image => chase" survives any feedback.
        self.aug_gap_k = float(aug_gap_k)
        # Deterministic enumeration for non-augmented sets (val/test): the
        # random row sampler made the val loss a fresh Monte-Carlo draw on
        # every pass, and the best-val checkpoint star is decided on it - noisy
        # model selection.  _flat covers every valid row exactly
        # once, so the val loss is deterministic and comparable across epochs
        # and runs.  Training keeps the phase-pool sampler (by design).
        self._flat = None
        if not self.augment:
            self._flat = [(k, j) for k, e in enumerate(self.eps)
                          for j in range(self.stack - 1, e.n)]

    def __len__(self):
        return sum(e.n - self.stack + 1 for e in self.eps)

    def __getitem__(self, i):
        n = self.stack
        if self._flat is not None:
            k, j = self._flat[i]
            e = self.eps[k]
        else:
            e = self.eps[i % len(self.eps)]
            r = np.random.rand()
            if r < self.desc_p and len(e.desc):
                pool = e.desc
            elif r < self.desc_p + self.trans_p and len(e.trans):
                pool = e.trans
            else:
                pool = (e.moving if (np.random.rand() < P_MOVE
                                     and len(e.moving)) else e.still)
            if not len(pool):
                pool = np.arange(n - 1, e.n)
            j = int(pool[np.random.randint(len(pool))])
        # newest first, uniform 100 ms spacing (matches evaluate.py)
        stack = e.g[j - n + 1:j + 1][::-1].astype(np.float32) / 255.0
        # proprio = IST || SOLL.  The order is load-bearing: evaluate.py builds
        # the same vector the same way, and swapping the halves would have the
        # policy read its commands as its state.
        prop = np.concatenate([e.q[j], e.soll[j]]).astype(np.float32)
        if self.augment:
            dx, dy = np.random.randint(-4, 5, 2)
            # pad + crop, NOT np.roll: roll wraps the opposite edge's pixels
            # into view (a table corner cannot materialise in the sky), edge
            # replication is the honest shift.  Same shift direction
            # as roll, so the rotation grid's fx/fy compensation below is
            # unchanged.
            if dx or dy:
                stack = np.pad(stack,
                               ((0, 0), (max(0, dy), max(0, -dy)),
                                (max(0, dx), max(0, -dx))), mode="edge")
                r0, c0 = max(0, -dy), max(0, -dx)
                stack = stack[:, r0:r0 + D_RES, c0:c0 + D_RES]
            # SOLL is the commanded target: in the rollout it jitters with the
            # policy's own output noise, so it gets the same sigma as IST.
            if self.aug_sigma > 0.0:
                prop = prop + np.random.randn(12).astype(np.float32) \
                    * self.aug_sigma
            if self.aug_gap_k > 0.0:
                # scale the SOLL-IST GAP, not the two halves independently:
                # independent noise shrinks/roughens the gap but never produces
                # the 0.3-1.2 rad band the policy's own feedback creates
                gap = prop[6:12] - prop[0:6]
                prop[6:12] = prop[0:6] + gap * np.float32(
                    np.random.uniform(0.0, self.aug_gap_k))
            if self.aug_drop_p > 0.0:
                # proprio dropout (copycat remedy, train only): zero the
                # whole IST||SOLL vector with prob p (no rescaling:
                # scaling joint angles would fabricate false states).
                # E2 measured 8.7x proprio reliance; dropped rows must be
                # solved from vision alone. Eval (augment=False) always
                # sees full proprio.
                if np.random.rand() < self.aug_drop_p:
                    prop = np.zeros_like(prop)
            if self.aug_rot_deg > 0.0:
                # a wrist offset shows up in a wrist-mounted camera as a
                # rotation of the whole view; +-1.2 deg is 2x the rollout's
                # worst 0.006 rad (= 0.34 deg) excursion, sampled uniformly
                # so the policy has a basin rather than a knife edge
                th = float(np.deg2rad(np.random.uniform(
                    -self.aug_rot_deg, self.aug_rot_deg)))
                c, s = np.cos(th), np.sin(th)
                # one matrix for the whole stack: the arm is rigid over 280 ms
                m = np.array([[c, -s], [s, c]], np.float32)
                # torch grid_sample: coordinates in [-1, 1]; the roll above
                # shifted the content by (dx, dy) px, so rescale to match
                yy, xx = np.meshgrid(
                    np.linspace(-1, 1, D_RES, dtype=np.float32),
                    np.linspace(-1, 1, D_RES, dtype=np.float32), indexing="ij")
                fx = np.float32(1 + float(dx) / (D_RES / 2.0))
                fy = np.float32(1 + float(dy) / (D_RES / 2.0))
                gx = (m[0, 0] * xx + m[0, 1] * yy) / fx
                gy = (m[1, 0] * xx + m[1, 1] * yy) / fy
                grid = torch.from_numpy(np.stack([gx, gy], -1))[None].expand(
                    n, -1, -1, -1)
                stack = torch.nn.functional.grid_sample(
                    torch.from_numpy(stack)[:, None], grid,
                    mode="bilinear", padding_mode="border",
                    align_corners=False)[:, 0].numpy()
        return (torch.from_numpy(np.ascontiguousarray(stack)),
                torch.from_numpy(prop),
                torch.from_numpy(e.chunks[j]))


def make_loaders(ds, cache_dir, batch=256, workers=0, stack=None, extra=None):
    """extra: one (ds, cache) pair or a LIST of them.

    A single pair keeps the round-1 shape (ds_0010_d1, cache_0010_d1); a list
    joins several DAgger rounds (d1 AND d2) in one --extra-ds argument so
    round-1 and round-2 corrections train together while val/test stay the
    untouched ds split.  Duplicate (ds, cache) pairs are dropped so the same
    round cannot be added twice; the in-memory episodes stay shared, so all
    training episodes sit in one DataLoader dataset (only the shuffle rng is
    shared across the extras - harmless for BC).
    """
    import ramcap
    # ds and cache_dir are REQUIRED and checked against each other: the old
    # defaults (ds="ds_0001", dir="cache") silently trained a ds_0003 run on
    # the ds_0002 cache, which reads perfectly and is simply the wrong data.
    rows = load_manifest(ds)
    print(f"data: {ds} + {cache_dir}/")
    tr_ids, va_ids, te_ids = split_ids(rows)
    tr = BCDataset(tr_ids, stack=stack, cache_dir=cache_dir, ds=ds,
                   aug_sigma=AUG_SIGMA, aug_rot_deg=AUG_ROT_DEG,
                   aug_gap_k=AUG_GAP_K, aug_drop_p=AUG_DROP_P,
                   trans_p=P_TRANS, desc_p=P_DESC)
    if extra:
        # DAgger corrections: episodes the POLICY's own states, labelled by the
        # expert (dagger_collect.py).  They join TRAIN ONLY - val/test stay the
        # untouched ds split so the val metric stays comparable across runs.
        pairs = ([extra] if isinstance(extra[0], str) else list(extra))
        seen = set()
        n_added = 0
        for eds, ecache in pairs:
            if (eds, ecache) in seen:
                continue
            seen.add((eds, ecache))
            erows = load_manifest(eds)
            # ok-only, like the base contract: the collector writes npz=None
            # for failed episodes so they never train; dagger_collect keeps
            # failed episodes' npz (drift + failure labels), and imitating
            # a failed demo teaches failure. Drifted-but-SUCCESSFUL
            # episodes (the actual corrections) still join.
            nok = sum(1 for r in erows if r.get("npz") and not r.get("ok"))
            eids = sorted(int(r["ep_id"]) for r in erows
                          if r.get("npz") and r.get("ok", True))
            if nok:
                print(f"  ({eds}: skipping {nok} failed episodes; "
                      f"only successful demos train)")
            want = os.path.basename(os.path.normpath(eds))
            for i in eids:
                ep = CacheEp(i, ecache)
                assert ep.ds == want, \
                    f"{ecache}/ep_{i:06d}.npz tagged {ep.ds}, {eds}"
                ep._stk = tr.stack
                ep._split()
                tr.eps.append(ep)
                n_added += 1
            print(f"DAgger corrections: +{len(eids)} episodes from {eds} + "
                  f"{ecache} added to train only (val/test stay {ds})")
        print(f"DAgger total: +{n_added} correction episodes")
    va = BCDataset(va_ids, augment=False, stack=stack, cache_dir=cache_dir,
                   ds=ds)
    te = BCDataset(te_ids, augment=False, stack=stack, cache_dir=cache_dir,
                   ds=ds)
    print(f"manifold width: +-{tr.aug_rot_deg} deg wrist rotation, "
          f"+-{tr.aug_sigma} rad IST/SOLL (uniform over rows), "
          f"gap scale U(0, {tr.aug_gap_k:g}), proprio-drop p={tr.aug_drop_p:g}, "
          f"trans-sample p={tr.trans_p:g}")
    print(f"stack {tr.stack} frames ({(tr.stack - 1) * GRID_MS} ms at "
          f"{1000 // GRID_MS} Hz)  "
          f"episodes: train {len(tr.eps)}  val {len(va.eps)}  "
          f"test {len(te.eps)}  samples: train {len(tr)}  val {len(va)}")
    # Worker budget.  Count only what a worker holds PRIVATELY: with
    # a shared memmap the 1.18 GB of frames are page-cache
    # backed and shared, and what is left per worker is the proprio/chunk
    # arrays (~27 MB for 243 episodes).  Without the memmap each worker also
    # pays the full image array, which is what froze the machine.
    per_worker = sum((e.n * 12 + e.n * 8 * 6) * 4 for e in tr.eps)
    shared = all(getattr(e, "_fr", None) is not None for e in tr.eps)
    if not shared:
        per_worker += sum(e.n * 96 * 96 for e in tr.eps)
    allowed = ramcap.workers_for(per_worker, fraction=0.35)
    workers = max(0, min(int(workers), allowed))
    if workers and ramcap.MODE == "job" and ramcap._capped:
        # A child created inside the job object fails to allocate during its
        # own start-up (measured: MemoryError in reduction.dump, then EOFError
        # in the child).  Load in-process instead: it is ~2x slower per epoch
        # and uses no extra process at all, which is the trade the cap is for.
        print("loader: forcing 0 workers - spawned workers cannot start "
              "inside the OS RAM cap (set SMARTCELL_RAM_CAP_MODE=cooperative "
              "or SMARTCELL_RAM_CAP_GB higher to allow them)")
        workers = 0
    ramcap.report()
    print(f"loader: {workers} worker(s) of {per_worker / 2**30:.3f} GB each "
          f"({'shared frame memmap' if shared else 'per-worker image copies'}"
          f", cap {ramcap.CAP_GB:.1f} GB)")
    # val runs IN-PROCESS: the hardcoded num_workers=1 spawned a
    # worker per epoch, bypassing the RAM-cap worker policy above - under
    # memory pressure that spawn is the one that dies first.
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True,num_workers=workers, drop_last=True,
                                            persistent_workers=workers > 0),
            torch.utils.data.DataLoader(va, batch_size=batch, num_workers=0),
            te)


if __name__ == "__main__":
    import sys
    import time
    ds = sys.argv[1] if len(sys.argv) > 1 else "ds_0003"
    cd = sys.argv[2] if len(sys.argv) > 2 else f"cache_{ds[-4:]}"
    tr, va, te = make_loaders(ds, cd)
    img, q, act = next(iter(tr))
    print("batch:", tuple(img.shape), tuple(q.shape), tuple(act.shape),
          f"img [{img.min():.2f},{img.max():.2f}]")
    assert img.shape[1:] == (tr.dataset.stack, RES, RES) and act.shape[1:] == (CHUNK, 6)
    t0 = time.perf_counter()
    for _ in range(20):
        next(iter(tr))
    print(f"loader throughput: {20 * 256 / (time.perf_counter() - t0):.0f} samples/s")
    print("OK")
