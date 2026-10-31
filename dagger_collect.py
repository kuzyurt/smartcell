"""DAgger corrections: the expert labels the states the POLICY creates.

WHY (measured, Act 15/16 of assets/PROVENANCE.md):
  The closed loop drifts off the recorded wait pose (camera feedback loop,
  probe_wobble_source.py) and lands in the SOLL-IST "dead band" at the chase
  transition (probe_gap_location.py).  Both are states that occur NOWHERE in
  ds_0010, so the model improvises there: wobble at the wait, retreat at the
  chase.  Noise augmentation cannot guarantee coverage of the states the
  policy itself produces.  DAgger does, by construction: visit the state, ask
  the expert what to do, train on the pair.

HOW (the command-walk, not a state jump):
  The arm is a stiff position servo - the state tracks the COMMAND within a
  grid step (measured: a state kick of 0.033 rad survives ~2 ms).  The policy's
  wobble is therefore a COMMAND wander, and the only physically consistent way
  to visit "arm 0.15 rad off the wait pose" is to COMMAND off the wait pose.
  So during selected phase windows the APPLIED command random-walks around the
  expert's target (the arm follows, the wrist camera shifts exactly as the
  policy's own drift shifts it), while the LABEL stays the expert's INTENDED
  command.  The pair is then: "state where the policy wobbles -> expert action
  that fixes it (return to the wait pose / keep dashing)".

  The cache format separates the two natively (build_cache.py):
    act_raw  the raw actuator command   -> soll (what the net reads as SOLL)
    act      the normalised label       -> chunks (what the net must output)
  The original collector wrote the same command into both; here they differ on
  purpose: act_raw = applied (walked) command, act = expert's intended.

  Frames are rendered LIVE in the hook: render_dataset.py re-drives from
  act_raw and would follow the WALKED commands instead of the recorded states
  (its replay loop steps act_raw), so the correction frames cannot come from
  it.  Live capture reproduces its conventions exactly: 128x128 RGB wrist_cam,
  mount + VisApply per-episode visual jitter, frame at obs t captured
  u~U(0, 40 ms) stale (img_t vs img_row/latency), 25 Hz grid.

WINDOWS (episode-relative seconds): two wait walks, one dash/chase walk with
  bigger steps (bigger SOLL-IST gaps with chase labels), one carry/place walk,
  one return walk.  A window landing in another phase is harmless: the labels
  are the expert's own commands, valid in every phase.

  ROUND 2 (v6) retargeted from the measured v5 failures (trace_v5.json,
  evaluate.py --trace, probe_failure_timing.py), not from the round-1 guesses:      * the close is commanded on schedule (~9.5 s) in EVERY rollout, but the
      site trails the part by 48-78 mm in x when it fires - the jaws shut in
      the air while the arm is still chasing (grasp succeeds only when the
      still-running servo snags the part ~0.6 s later).  The walk must reach
      INTO the late chase, up to but not through the expert's close: a walked
      gap that survives the close breaks the grasp, truncates the episode and
      deletes the carry labels - the late-chase window therefore ENDS before
      the close (t1<=10.2 s, close ~10.2-10.6 s).
      MEASURED WRONG in hindsight (probe_d2_windows.py): the intended
      "latch off at the close" compared grip <= 0.0, but the expert's close
      command is GRIP_SHUT = 0.02 > 0, so the latch NEVER fired (d2's
      carry/descent walks ran at full bound through the close), and the walk
      sat on 1.9% of the pre-close chunk rows (gap up to 0.374 rad) with the
      whole final approach displaced (tcp->part median 27.3 mm vs 15.9 mm
      walk-free) - the "close while not at the part" supervision that v6
      learned as close-on-air (0/72 picks, see Act 18).
    * after the grasp the carry wastes ~3 s going the WRONG way (+x with the
      part in the jaws) before turning to the tote, then hovers over the tote
      at LIFT_Z and never descends to release within the 20 s horizon.  The
      carry walk is extended over the wander (11.2-14.6 s) and the descent
      walk is enlarged (14.6-17.8 s), so the off-manifold carry/descent states
      get expert labels (descend and release) too.
  Round-1 windows are otherwise unchanged: the wait-wobble fix measurably
  worked (probe_wobble_source.py on v5: ist_drift 0.034 rad vs v3's 0.103).

  ROUND 3 (v7) fixes the measured round-2 regression:
    * pre-close walks (W1-W4) FREEZE at the expert's final grasp descent
      (move_to z == GRASP_Z, WalkHook.set_descent) and at the close command:
      every label chunk in the last 0.32 s before the close contains the
      close, so those rows must be walk-free.
    * carry/descent walks (W5/W6) are POST-GRASP GATED: they only start
      GRASP_SETTLE_S after the close command (close sims run 0.8 s, then the
      lift - a walk there breaks the grasp and the episode keeps
      "carry with empty jaws" rows, which is v6's second failure mode).
    * the late chase is tamed (bound 0.4 -> 0.25, step 0.14 -> 0.10).

FAILED EPISODES ARE KEPT: the (state, expert label) pairs are valid regardless
  of task outcome, and a failed grasp is exactly the recovery situation that
  is missing from ds_0010.

Run:  python dagger_collect.py --n 64 --out ds_0010_d1
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import mujoco

import ramcap          # HARD 5 GB cap, see ramcap.py

import build_cell as C
import cam_variant as V
import demo_pick
import expert as E
import current
import collect_state as CS
from collect_state import RandEnv
from conveyor_env import ConveyorPickEnv, GRIP_SHUT
from render_dataset import sample_vis, VisApply

RES = 128
FPS = 25
DT = 0.005
# Post-grasp gate: the expert's close sims run 0.5 + 0.3 s, then the lift.
# A carry walk during either breaks the grasp; start walking only once the
# part is held and lifted (measured v5 slips: 3 at lift-off, 10 mid-carry).
GRASP_SETTLE_S = 1.5
LAT_MAX = 1.0 / FPS
GRID_STEPS = 8                 # physics steps per 40 ms grid tick
N_MAX = FPS * 32               # frame slots preallocated; truncated at the end


class WalkHook:
    """make_hook's recorder + command-walk + live frames + label capture.

    Called once per physics step with the POST-step state.  Per call:
      1. record the APPLIED pair (post-step state, d.ctrl)  -> act_raw
      2. record the INTENDED command of the current sim call -> act (label)
      3. advance the walk offset at 40 ms ticks (sim_annotated reads it via
         offset_for() and commands target+offset, so the walk is part of the
         physics rather than a post-hoc state jump)
      4. render if an obs row is due.
    """

    def __init__(self, buf, windows, rows_want, lat, mount, rend, cam_id,
                 rng, arm_aid):
        self.rec = CS.make_hook(buf)
        self.windows = windows
        self.rows_want = rows_want
        self.lat = lat
        self.mount = mount
        self.rend = rend
        self.cam_id = cam_id
        self.rng = rng
        self.arm_aid = arm_aid
        self.calls = 0
        self.done = 0
        self.imgs = np.empty((len(rows_want), RES, RES, 3), dtype=np.uint8)
        self.last_tick = -1.0
        self.X = None               # the expert's intended arm target
        self.grip = None            # ... and grip command
        self.intended = []          # per recorded step: the label command
        self.active = None          # the window being walked, if any
        self.pre_frozen = False     # final approach began: pre-close walks off
        self.close_t = None         # sim time of the expert's close command

    def set_intended(self, arm, grip, t=None):
        """Wrap point: every sim() call announces the command it is driving."""
        if grip is not None:
            self.grip = float(grip)
            if self.close_t is None and self.grip <= GRIP_SHUT + 1e-6:
                # The expert's close command.  NOTE: GRIP_SHUT = 0.02 > 0 -
                # the old `grip <= 0.0` latch therefore NEVER fired (measured:
                # d2's walks ran at full bound through the close and the
                # whole carry).  Freeze the pre-close walks; the post-grasp
                # walks open after GRASP_SETTLE_S (see _walk_on).
                self.pre_frozen = True
                self.close_t = float(t) if t is not None else 0.0
        if arm is not None:
            self.X = np.asarray(arm, dtype=np.float64).copy()

    def set_descent(self, target):
        """The expert's FINAL GRASP DESCENT has begun (move_to z == GRASP_Z).

        From here to the close, every 0.32 s label chunk contains the close
        command: walking those rows teaches "close while not at the part".
        Measured (probe_d2_windows.py): d2's pre-close chunk rows carry walk
        gaps up to 0.374 rad and a displaced approach (tcp->part median
        27.3 mm vs 15.9 mm walk-free) - and v6 (trained on d2) closes on air
        22-62 mm from the part.  Freeze the pre-close walks from this move.
        """
        try:
            z = float(np.asarray(target, dtype=np.float64).ravel()[2])
        except (TypeError, ValueError):
            return
        if abs(z - demo_pick.GRASP_Z) < 1e-6:
            self.pre_frozen = True

    def _walk_on(self, w, t):
        """Whether window w may walk at sim time t (the phase gate)."""
        if "protect" not in w:
            return True             # v1/v2 replay: legacy semantics unchanged
        if w["protect"]:
            return not self.pre_frozen
        # post-grasp window: only once the grasp is established and lifted.
        # v5 early-lift windows may ask for a shorter settle (their whole
        # point is to reach the lift-start), but the default stays 1.5 s.
        return (self.close_t is not None
                and t >= self.close_t + w.get("settle", GRASP_SETTLE_S))

    def offset_for(self, t):
        """The walk offset to ADD to the commanded target at sim time t.

        The walk lives HERE (applied to the arm= argument of sim) and not in
        the hook: expert's wait loop drives one physics step per sim() call,
        and every call re-sets ctrl to its own arm target - a hook-side bend
        of d.ctrl is overwritten before the next step ever runs (measured:
        zero state drift during the wait windows).  Commanding target+off
        instead makes the arm genuinely walk: the state follows the servo,
        the wrist camera shifts exactly as the policy's drift shifts it.
        Pre-close walks freeze at the final grasp descent and at the close;
        post-grasp walks open GRASP_SETTLE_S after the close (see _walk_on).
        """
        for w in self.windows:
            if w["t0"] <= t <= w["t1"] and self._walk_on(w, t):
                return w["off"]
        return None

    def __call__(self, env):
        r = env.real
        self.calls += 1
        self.rec(env)                       # (state, applied ctrl) -> act_raw
        self.intended.append(np.append(self.X, self.grip))   # -> act label

        now = r._sim_t
        win = None
        for w in self.windows:
            if w["t0"] <= now <= w["t1"] and self._walk_on(w, now):
                win = w
                break
        self.active = win
        if win is not None and now - self.last_tick >= GRID_STEPS * DT - 1e-9:
            win["off"] = np.clip(
                win["off"] + self.rng.uniform(-win["step"], win["step"], 5),
                -win["bound"], win["bound"])
            self.last_tick = now

        while self.done < len(self.rows_want) and \
                self.calls >= self.rows_want[self.done] + 1:
            self.mount.update(r.data)
            self.rend.update_scene(r.data, camera=self.cam_id)
            V.strip_scene_markers(self.rend)
            self.imgs[self.done] = self.rend.render()
            self.done += 1


def schedule(rng, walk_version=2):
    """(t0, t1, step, bound) command-walk windows, one per failure phase.

    step = random-walk step per 40 ms grid tick, bound = clamp around the
    expert's own target.  The dash window uses bigger steps on purpose: a
    faster walk leaves the stiff servo lagging 0.1-0.3 rad behind, which is
    exactly the SOLL-IST gap band the policy's lunge dies in (Act 15).

    v2 (round 2) retargets the late windows at the measured v5 failures: a
    late-chase walk that reaches up to the expert's close, a carry walk over
    the measured +x wander (t=12-14) and a descent walk over the measured
    no-descent hover (t=15-18) that trace_v5.json actually shows.  The two
    chase walks were MEANT to latch off at the expert's close command, but
    that latch compared grip <= 0.0 and the close command is GRIP_SHUT = 0.02:
    it never fired (measured on d2 - see WalkHook.set_intended).  v3 (round 3)
    replaces the latch with a real phase gate and tames the late chase; v1/v2
    are kept for provenance replay and keep their exact legacy semantics
    (windows without a 'protect' key are never gated).

    protect=True   pre-close walk: frozen from the final grasp descent
                   (set_descent) and from the close command on.
    protect=False  post-grasp walk: only walks >= GRASP_SETTLE_S after the
                   close command, so the close/lift never sees a walked SOLL.
    """
    def U(a, b):
        return float(rng.uniform(a, b))

    if walk_version <= 1:
        return [
            dict(t0=U(1.0, 5.5), t1=U(6.0, 7.2), step=0.06, bound=0.25,
                 off=np.zeros(5)),              # wait wobble (the wobble fix)
            dict(t0=U(6.3, 7.8), t1=U(7.4, 8.6), step=0.12, bound=0.35,
                 off=np.zeros(5)),              # big-gap wait states
            dict(t0=U(8.9, 9.8), t1=U(9.7, 10.3), step=0.12, bound=0.30,
                 off=np.zeros(5)),              # dash/chase (the retreat fix)
            dict(t0=U(11.8, 14.2), t1=U(12.8, 15.2), step=0.08, bound=0.25,
                 off=np.zeros(5)),              # carry/place
            dict(t0=U(15.9, 17.2), t1=U(16.9, 18.2), step=0.06, bound=0.25,
                 off=np.zeros(5)),              # return to start
        ]
    if walk_version == 3:
        # ROUND 3 (v7): fix the measured round-2 regression (v6 = 0/72 picks,
        # closes on air 22-62 mm from the part).  The pre-close chunk rows
        # (every row whose 0.32 s label chunk contains the close) must be
        # walk-free - hence the phase gate in WalkHook, not tighter end times:
        # the close lands anywhere in 10.3-12.1 s (measured on d2), so no
        # fixed t1 can both reach into the chase and stay 0.32 s clear of it.
        return [
            dict(t0=U(1.0, 5.5), t1=U(6.0, 7.2), step=0.06, bound=0.25,
                 off=np.zeros(5), protect=True),      # wait wobble (the wobble fix)
            dict(t0=U(6.3, 7.8), t1=U(7.4, 8.6), step=0.12, bound=0.35,
                 off=np.zeros(5), protect=True),      # big-gap wait states
            dict(t0=U(8.9, 9.8), t1=U(9.7, 10.2), step=0.12, bound=0.30,
                 off=np.zeros(5), protect=True),      # dash/chase (the retreat fix)
            dict(t0=U(9.1, 9.9), t1=U(9.9, 10.2), step=0.10, bound=0.25,
                 off=np.zeros(5), protect=True),      # late chase (tamed 0.4->0.25)
            dict(t0=U(11.5, 12.2), t1=U(13.8, 14.6), step=0.08, bound=0.25,
                 off=np.zeros(5), protect=False),     # carry: covers the +x wander
            dict(t0=U(14.6, 15.2), t1=U(16.8, 17.8), step=0.10, bound=0.30,
                 off=np.zeros(5), protect=False),     # descent/release (v5 never descends)
        ]
    if walk_version == 4:
        # ROUND 4: retimed for the no-hover expert (flat carry waypoint +
        # 0.5 s settle removed - the hover it taught is the measured 0033
        # stall: 15/20 place failures hover 8-57 mm over the tote).  Measured
        # on d21 (first fixed-expert round): close med 10.84 s, release med
        # 14.56 s, carry+descent 3.7 s.  The v3 descent window (t0 14.6-15.2)
        # lands ON/after the release - it walks the return, not the descent.
        # v4 walks the transit + release legs (12.6-15.8) plus a late-descent
        # tail (13.4-16.0) over the old hover territory, so jittered
        # near-hover states get descend/release labels instead of
        # return-to-hover ones.  Pre-close windows unchanged from v3.
        return [
            dict(t0=U(1.0, 5.5), t1=U(6.0, 7.2), step=0.06, bound=0.25,
                 off=np.zeros(5), protect=True),      # wait wobble (the wobble fix)
            dict(t0=U(6.3, 7.8), t1=U(7.4, 8.6), step=0.12, bound=0.35,
                 off=np.zeros(5), protect=True),      # big-gap wait states
            dict(t0=U(8.9, 9.8), t1=U(9.7, 10.2), step=0.12, bound=0.30,
                 off=np.zeros(5), protect=True),      # dash/chase (the retreat fix)
            dict(t0=U(9.1, 9.9), t1=U(9.9, 10.2), step=0.10, bound=0.25,
                 off=np.zeros(5), protect=True),      # late chase (tamed 0.4->0.25)
            dict(t0=U(11.8, 12.4), t1=U(13.8, 14.6), step=0.08, bound=0.25,
                 off=np.zeros(5), protect=False),     # carry/transit
            dict(t0=U(12.6, 13.2), t1=U(14.8, 15.8), step=0.10, bound=0.30,
                 off=np.zeros(5), protect=False),     # descent (transit+release)
            dict(t0=U(13.4, 14.0), t1=U(15.2, 16.0), step=0.12, bound=0.35,
                 off=np.zeros(5), protect=False),     # late-descent tail
        ]
    if walk_version == 5:
        # ROUND 5: v4's windows plus the measured gap.  Fresh-seed traces
        # (trace_0037d_9999.geo) show the picked-not-placed losses happen
        # 0.08-1.7 s AFTER the close, at lift z 0.64-0.85 - inside the
        # close+GRASP_SETTLE_S(1.5) blind spot, because any walked lift under
        # the old expert broke the grasp.  Under the no-hover expert a SMALL
        # early-lift window (bound 0.10 rad, settle 0.6 s) teaches recovery
        # from lift-start wobble while (hopefully) not shedding the part;
        # episodes it does break are skipped at train like any other fail.
        w = schedule(rng, 4)
        w.append(dict(t0=U(10.2, 10.8), t1=U(12.2, 12.8), step=0.04,
                      bound=0.10, off=np.zeros(5), protect=False, settle=0.6))
        return w
    return [
        dict(t0=U(1.0, 5.5), t1=U(6.0, 7.2), step=0.06, bound=0.25,
             off=np.zeros(5)),                  # wait wobble (the wobble fix)
        dict(t0=U(6.3, 7.8), t1=U(7.4, 8.6), step=0.12, bound=0.35,
             off=np.zeros(5)),                  # big-gap wait states
        dict(t0=U(8.9, 9.8), t1=U(9.7, 10.2), step=0.12, bound=0.30,
             off=np.zeros(5)),                  # dash/chase (the retreat fix)
        dict(t0=U(9.1, 9.9), t1=U(9.9, 10.2), step=0.14, bound=0.40,
             off=np.zeros(5)),                  # late chase (latched off at close)
        dict(t0=U(11.2, 12.0), t1=U(13.8, 14.6), step=0.08, bound=0.25,
             off=np.zeros(5)),                  # carry: covers the +x wander
        dict(t0=U(14.6, 15.2), t1=U(16.8, 17.8), step=0.10, bound=0.30,
             off=np.zeros(5)),                  # descent/release (v5 never descends)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="ds_0011_d1")
    ap.add_argument("--ds", default="ds_0011",
                    help="base dataset whose line record configures the belt "
                         "and whose variance contract this round follows "
                         "(default ds_0011: narrow belt var + wait jitter)")
    ap.add_argument("--no-wait-jitter", action="store_true",
                    help="exact wait pose every episode (reproduce ds_0010 rounds)")
    ap.add_argument("--base-seed", type=int, default=210000,
                    help="disjoint from ds_0010 (100000+) and eval (10000+)")
    ap.add_argument("--walk-version", type=int, default=3,
                    choices=(1, 2, 3, 4, 5),
                    help="1 = round-1 windows (ds_0010_d1), 2 = round-2 "
                         "windows (ds_0010_d2, regression - kept for "
                         "replay), 3 = round-3 phase-gated windows, "
                         "4 = round-4 windows retimed for the no-hover "
                         "expert (d22+), 5 = v4 + small early-lift window "
                         "(the close+1.5 s blind spot)")
    a = ap.parse_args()

    ramcap.install()
    os.makedirs(os.path.join(a.out, "imgs"), exist_ok=True)
    man = os.path.join(a.out, "manifest.jsonl")

    # the wait pose and the line config are part of the episode contract:
    # same wait joints and same running-line record as --ds (default ds_0011)
    CS.WAIT_JOINTS = list(current.WAIT_JOINTS)
    # no tote-look legs here (see CS.TOTE_LOOK): the walk's drift offsets
    # and second-accurate phase windows are incompatible with the look
    # centering + hold. Dagger rounds correct drift; the look pattern
    # comes from the dedicated look dataset.
    CS.TOTE_LOOK = False
    rows = [l for l in open(f"{a.ds}/manifest.jsonl") if l.strip()]
    CS.apply_line_config(json.loads(rows[0]).get("line"))
    print(f"line ({a.ds}):", "RUNNING" if CS.RUNNING_LINE else "stopping",
          "belt var" if CS.BELT_NARROW_VAR else "fixed/broad")

    env = RandEnv(ConveyorPickEnv())
    r = env.real
    visap = VisApply(r.model)
    # shared mount (see CS.get_mount): the look centering inside
    # episode_stable_wait needs the same Mount - a second V.Mount(m)
    # trips check_mount because the first one world-parents the camera
    mount = CS.get_mount(r.model)
    rend = mujoco.Renderer(r.model, height=RES, width=RES)
    rend._scene_option.geomgroup[:] = [1] * 6
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    cam_id = int(r.model.camera("wrist_cam").id)
    codeh = CS.code_hash()

    # announce every sim call's intended command to the hook (outermost wrap:
    # collect_state's _patched_sim is E.sim, and expert.move_to resolves `sim`
    # through the module at call time, so this sees EVERY drive, incl. legs)
    _sim = E.sim
    _move_to = E.move_to

    def sim_annotated(env_, seconds, arm=None, grip=None, hook=None):
        h = hook if hook is not None else CS.CTX.get("hook")
        if h is not None and hasattr(h, "set_intended"):
            h.set_intended(arm, grip, float(env_.real._sim_t))  # the LABEL
            if arm is not None:
                off = h.offset_for(float(env_.real._sim_t))
                if off is not None:        # the APPLIED command walks
                    arm = np.asarray(arm, dtype=np.float64) + off
        return _sim(env_, seconds, arm=arm, grip=grip, hook=hook)

    E.sim = sim_annotated

    # The expert's final grasp descent is the walk's stop signal: move_to is
    # called with Cartesian targets, so z == demo_pick.GRASP_Z identifies the
    # descent-to-grasp move (the tote release uses TOTE_RELEASE_Z, and a
    # false trigger after the close is harmless).  Installed OUTSIDE
    # collect_state's own move_to patch so the target is the pre-shift one.
    def move_to_annotated(env_, target, grip, *a, **k):
        h = CS.CTX.get("hook")
        if h is not None and hasattr(h, "set_descent"):
            h.set_descent(target)
        return _move_to(env_, target, grip, *a, **k)

    E.move_to = move_to_annotated

    n_ok = 0
    for ep in range(a.n):
        seed = a.base_seed + ep
        rng = np.random.default_rng(seed ^ 0x5A5A)
        env.vec = CS.sample_vector(seed)
        if a.no_wait_jitter:
            env.vec["wait_joints"] = [float(x) for x in current.WAIT_JOINTS]
        else:
            vrng = np.random.default_rng(
                (int(seed) ^ 0x51F7 ^ CS.SEED_SALT) & 0xFFFFFFFF)
            env.vec["wait_joints"] = CS.sample_wait_joints(
                current.WAIT_JOINTS, vrng)
        CS.CTX.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                       for k, v in env.vec.items()})

        # per-episode visual jitter: what render_dataset applied to ds_0010
        vis = sample_vis(np.random.default_rng(seed + 7919))
        visap.apply(vis)
        mount.back = V.DEFAULT["back"] + vis["dback"]
        mount.deg = V.DEFAULT["deg"] + vis["ddeg"]
        mount.azim = V.DEFAULT["azim"] + vis["dazim"]

        # obs grid + per-frame latency, render_dataset's formula exactly
        lat = rng.uniform(0.0, LAT_MAX - 1e-4, N_MAX)
        rows_want = np.round((np.arange(N_MAX) / FPS - lat) / DT).astype(int)
        rows_want = np.clip(rows_want, 0, None)
        for j in range(1, N_MAX):
            if rows_want[j] <= rows_want[j - 1]:
                rows_want[j] = rows_want[j - 1] + 1

        buf = dict(t=[], qpos=[], grip=[], tcp=[], cubes=[], active=[],
                   act_raw=[], row0=None, hash=hashlib.sha256())
        hook = WalkHook(buf, schedule(rng, a.walk_version), rows_want, lat,
                        mount, rend, cam_id, rng, r._arm_aid)
        CS.CTX["hook"] = hook
        t0 = time.perf_counter()
        idx, ok, stats = CS.episode_stable_wait(
            env, seed=seed, cfg=E.FAST, log=False, hook=hook,
            stats=dict(ik_s=0.0, solves=0, sim_s=0.0, steps=0, reseeds=0))
        wall = time.perf_counter() - t0
        CS.CTX.pop("hook", None)
        visap.restore()
        buf["row0"] = env.row0

        n_frames = hook.done
        reason = ("no part" if idx is None else
                  "grasp failed" if not ok and not r._in_tote(r._cube_pos(idx))
                  and r._grasped() != idx
                  else "not in tote" if not ok else "ok")
        row = dict(ep_id=ep, seed=seed, worker=0, ok=bool(ok),
                   picked=int(idx) if idx is not None else -1,
                   reason=reason,
                   vector={k: CS._round_vec(v) for k, v in env.vec.items()},
                   n_steps=len(buf["t"]), wall_s=round(wall, 2),
                   ik_s=round(stats["ik_s"], 2), solves=stats["solves"],
                   sim_s=round(stats.get("sim_time_s", 0.0), 2),
                   code=codeh, npz=None, line=CS.line_config(),
                   dagger=[dict(t0=round(w["t0"], 2), t1=round(w["t1"], 2),
                                step=w["step"], bound=w["bound"],
                                protect=w.get("protect"))
                           for w in hook.windows])
        # DAgger keeps FAILED episodes too: the (state, expert label) pairs
        # are valid regardless of task outcome, and a failed grasp is exactly
        # the recovery situation missing from ds_0010.  Only "no part" is
        # useless.
        if idx is not None and n_frames > 50:
            img_t = (np.arange(n_frames) / FPS).astype(np.float32)
            np.savez_compressed(
                os.path.join(a.out, "imgs", f"ep_{ep:06d}_img.npz"),
                img=hook.imgs[:n_frames], img_t=img_t,
                img_row=rows_want[:n_frames].astype(np.int32),
                latency=lat[:n_frames].astype(np.float32),
                vis=np.array(json.dumps(vis)))
            # NOTE: no replay_deviation check - act_raw is the WALKED command
            # stream and the recorded states match it, but the label act is
            # the expert's intended target.  A re-drive from act_raw would
            # reproduce the states; a re-drive from act would deviate by
            # design.  The pair (state, label) is what is kept.
            states = CS.states_array(buf)
            applied = np.asarray(buf["act_raw"], dtype=np.float64)
            intended = np.asarray(hook.intended, dtype=np.float64)
            assert len(intended) == len(applied), (len(intended), len(applied))
            act = CS.normalized_actions(env, intended)     # LABEL -> chunks
            name = f"ep_{ep:06d}.npz"
            np.savez_compressed(
                os.path.join(a.out, name), **states, act=act,
                act_raw=applied.astype(np.float32),       # APPLIED -> soll
                vector=json.dumps(row["vector"]), ep=np.int64(ep),
                seed=np.int64(seed))
            row["npz"] = name
            # load_manifest (dataset.py) keys on 'img'; ds_0010 rows carry the
            # imgs path written by render_dataset's pass - ours is written
            # live, same relative path
            row["img"] = f"imgs/ep_{ep:06d}_img.npz"
            n_ok += 1
        with open(man, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"ep {ep:3d}  {'OK ' if ok else 'FAIL'} {reason:14s} "
              f"frames {n_frames:4d}  wall {wall:5.1f}s  "
              f"({n_ok}/{ep + 1} kept)", flush=True)

    rend.close()
    env.real.close()
    print(f"done: {n_ok}/{a.n} episodes -> {a.out}/")


if __name__ == "__main__":
    main()
