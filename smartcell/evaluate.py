"""Wrist-policy evaluation on a continuously running MuJoCo conveyor.

Loads the trained policy and rolls it out in the real env, wrist-camera-only
obs (plus qpos), 8-step chunks executed open-loop for 2 steps at 25 Hz (80 ms
a decision instead of the 10 Hz run's 200 ms - the fast-motion fix).  Two
conditions:
  nominal   stock visuals, stock mount
  tier-B    fresh visual vector per episode (Phase 3 sampler incl. mount jitter)
Default place requires release and 0.4s settled in the tote. Use
--place-settle-s 0 for the historical tote-entry metric. Development episode
stream base is 4242; use a new --seed-base for independent confirmation.

Run:  python smartcell/evaluate.py --ckpt ckpt/policy_0041c.pt [--n 50]
"""
import argparse
import hashlib
import json
import time

# Install the single-process job cap before importing collect_state. That
# collector opts into cooperative mode for multiprocessing; importing it
# first previously disabled the evaluator's supposedly hard RAM cap.
import ramcap
import numpy as np
import mujoco
import torch

print("torch", torch.__version__)

import build_cell as C
import cam_variant as V
import expert as E
import collect_state as CS
from render_dataset import sample_vis, VisApply
from collect_state import RandEnv
from conveyor_env import ConveyorPickEnv, X_STOP
from model import Policy, load_state
from policy_runtime import (wrist_gray, GraspDwell, temporal_actions,
                            PlacementMonitor, ReleaseOnDescent, carry_lookahead)
import current

import os as _os

RES = 96
STACK = 8               # 8 frames at 40 ms = 280 ms of lookback
CHUNK = 8
EXEC = int(_os.environ.get("EXEC", "2"))
# execute 2 entries = 80 ms open loop (was 200 ms at 10 Hz)
CAM_DT = 0.04           # Phase 7: 25 Hz observation + decision grid (8 rows)
ENTRY_STEPS = 8         # physics steps per chunk entry = CAM_DT/DT
HORIZON_S = 20.0        # the policy drives the FULL episode from the wait pose
                        # (expert cycles are ~18 s; 20 s only bounds failures --
                        # successful rollouts break early on placed)
# The gripper channel is BISTABLE in every demonstration (the expert only ever
# commands 0.64 or -0.17453, i.e. -0.1515 / -1.0 normalised).  A regression head
# on a bistable target averages the transition: probe_rows.py measured the
# prediction as a RAMP over +-2 chunk entries, and probe_diverge6 caught the
# consequence - the close reached only -0.44 (43 mm of jaw gap, a 32 mm part)
# so the jaws shut in the air.  Decoding the channel by threshold is the
# correct decoder for a bang-bang actuator, not a smoothing hack: the close is
# then COMMANDED fully and, with the jaw-centring aim fix, stalls on the part.
GRIP_SNAP = True
GRIP_THRESH = -0.40
GRIP_OPEN_N, GRIP_SHUT_N = -0.1515, -1.0
# GRIP_SHUT_N is the actuator's real lower bound (-0.17453 rad raw): the new
# way's close command.  The old value here, -0.7974 = 0.02 rad raw, was the
# HALF-shut claw -- the servo stall point on a 32 mm part, not a close.  With
# it, an eval rollout could never fully close no matter what the policy said.
# Grasp primitive: after the close is commanded, the arm holds its last command
# for this long while the jaws shut - exactly what the demonstrations do
# (expert: sim(0.5, GRIP_SHUT) then sim(0.3, GRIP_SHUT) with the arm ctrl held
# at the grasp pose).  Without it the policy's own chunk can start the lift in
# the same 80 ms window as the close: measured, the grip shut at site z = +27
# mm above the part with the arm already rising, i.e. the jaws closed in the
# air.  The dwell only freezes the ARM; the grip channel is still the policy's.
# ACT 19 FIX: the old trigger (`hold_arm is None`) could only fire before the
# first executed step, so the dwell NEVER engaged - measured: 0 engagements in
# the 18,000 decision records of trace_v5.json.  Every eval number before
# Act 19 was therefore produced WITHOUT this primitive.  The trigger is now
# the open->shut EDGE of the commanded grip, so the policy's continuous shut
# commands during the carry cannot keep re-freezing the arm.
GRASP_DWELL_S = 0.40
DWELL_MODE = current.RUNTIME_PROFILE["dwell_mode"]
PLACE_SETTLE_S = current.RUNTIME_PROFILE["place_settle_s"]
RELEASE_PRIMITIVE = current.RUNTIME_PROFILE["release_primitive"]
RETAIN_ON_STALL = False
CARRY_LOOKAHEAD = 0
CARRY_HEAD = None
VISUAL_PICK = current.RUNTIME_PROFILE.get('visual_pick',False)
VISUAL_PLACE = current.RUNTIME_PROFILE.get('visual_place',False)
VISUAL_CLOSE_S = current.RUNTIME_PROFILE.get('visual_close_s',.64)
VISUAL_TAKEOVER_Z = current.RUNTIME_PROFILE.get('visual_takeover_z',.735)

# DIAGNOSTIC ONLY (--grip-gate-r): while the part is farther than this from
# the grasp site, force the grip channel open.  This is privileged state the
# policy does not get, so a gated run is NOT a measurement -- it isolates
# whether the chase/catch/carry work once the spurious early close (the
# probe_prime identical-stack failure) cannot kill the episode.
GRIP_GATE_R = None

ARM_PIN_S = float(_os.environ.get("ARM_PIN_S", "0.0"))
# ARM_MAX_DQ: rate-limit the executed arm command to this many rad from the
# CURRENT joint state, each 40 ms entry.  WHY: the expert's SOLL teleports
# 1.3-2.3 rad ahead of a motionless IST exactly at the wait->chase transition
# (probe_gap_location.py), then collapses under 0.3 within 6 rows.  The gap
# band 0.3-1.2 rad therefore NEVER occurs in training, and the policy's own
# 0.3 rad lunge lands the fed-back (IST, SOLL) pair in that valley -> the net
# retreats to the 230-row wait basin.  Clamping the executed command to the
# steady-chase envelope (expert p95 = 0.103 rad/step) keeps every fed-back
# pair on-manifold.  0 = off (default), e.g. ARM_MAX_DQ=0.12.
ARM_MAX_DQ = float(_os.environ.get("ARM_MAX_DQ", "0.0"))

# TE_M: ACT-style temporal ensembling.  At every decision the new 8-entry
# chunk votes on the entries it covers; overlapping chunks from earlier
# decisions keep voting with weight exp(-TE_M * age_in_decisions).  0 = off
# (bit-identical to the historical protocol).  WHY: the policy's within-chunk
# jerk is already 190x below the expert's (2.6e-5 vs 5.1e-3 squared norm), so
# the visible shiver is not noisy predictions - it is the chunk boundary, where
# a fresh 8-step plan can disagree with the 80 ms actually executed.  Only the
# arm channels are ensembled; the bistable grip must follow the NEWEST plan or
# the close would be averaged with an older open.
TE_M = float(_os.environ.get("TE_M", "0.0"))

# ARM_DEADBAND: latch the executed arm command while the new command stays
# within this many rad of the previous one.  The policy's parked-arm tremor is
# 0.008-0.011 rad per 40 ms entry (measured, _jitter_probe.py) with no intended
# motion; real chase/transport moves are 0.05-0.13 rad per entry, so a latch
# removes the shiver WITHOUT the ~100 ms plan lag that temporal ensembling
# introduced (TE at 0.1 lagged the chase by ~1.4 decisions and scored 0/5 -
# the grasp window is ~8 mm wide at belt speed).
#   ARM_DEADBAND  generic, every phase.  0.01 measured success-neutral
#                 (n=20: 90/85 vs base 95/80), but leaves HF amplitude.
#   ARM_PARK_DB   stronger latch used ONLY within ARM_PARK_R rad of the wait
#                 pose, where the visible shiver lives.  0.02 zeroes the parked
#                 HF command energy (0.0002 vs 0.073) and cannot touch the
#                 chase: at 0.02 everywhere pick fell 95% -> 70%.
ARM_DEADBAND = float(_os.environ.get("ARM_DEADBAND", "0.0"))
ARM_PARK_R = float(_os.environ.get("ARM_PARK_R", "0.0"))
ARM_PARK_DB = float(_os.environ.get("ARM_PARK_DB", "0.02"))

# ARM_EMA: first-order smoothing of the executed arm command,
# cmd = (1-a)*prev + a*new (0 = off).  0.5 halves the tremor at a 40 ms lag.
ARM_EMA = float(_os.environ.get("ARM_EMA", "0.0"))

# boxX: held-out box positions OUTSIDE the training range (collect_state:
# dx +-60 mm, dy -60..+20 mm), measured reachable for the expert at 5-9 mm
# drop error.  The fourth corner (-75, +25) mm is deliberately absent: it
# hits the arm's +y reachability wall (measured: 78 mm drop error, IK 84
# solves / 3 reseeds) and would test the arm, not the policy.  boxX REUSES
# nominal's RNG stream, so the episodes are the same and only the box moves -
# the comparison isolates box tracking.  Module level so watch.py uses the
# same list rather than a copy.
BOX_X_CORNERS = ((0.075, -0.075), (-0.075, -0.075), (0.075, 0.025))


def current_frame(env, mount, rend):
    mount.update(env.real.data)
    rend.update_scene(env.real.data, camera=int(env.real.model.camera("wrist_cam").id))
    V.strip_scene_markers(rend)      # the magenta light-debug sphere, see cam_variant
    # match cache/ep_*.npz exactly: area 128->96, gray = mean of the three
    # channels, uint8 quantisation (build_cache: np.rint(...).astype(uint8)),
    # then /255 like dataset.py.  Channel-0 without /255 = 255x OOD input ->
    # saturated policy output; without the rounding the eval input was a
    # smooth float where training saw 1/255 steps (Act 19 parity fix).
    return wrist_gray(rend.render())


def rollout(env, model, mount, rend, visap, vis, rng, stack=STACK, trace=None,
            lrng=None, box=None, on_step=None, stats=None):
    """on_step(dict) is called at every frame-capture boundary.

    Purely an observation hook for watch.py (the live viewer): when it is None
    - i.e. for every evaluation run in the provenance ledger - this function
    takes exactly the code path it always took.  Nothing below reads it, so
    results are unchanged by construction.
    """
    lrng = lrng if lrng is not None else rng
    dev = next((p.device for p in model.parameters()), torch.device("cpu"))
    r = env.real
    # eval vector = the collector's sampler (jaw yaw, aim jitter, hover dz,
    # grip depth), belt nominal.  The expert's IK branch choice depends on
    # the jittered target at the 5 mm level; eval MUST run the same code path
    # with the same channels or the handover pose lands on a branch the
    # policy never saw (measured: roll -0.007 in data vs -1.459 at eval ->
    # saturated OOD predictions).
    from collect_state import sample_vector as _sv
    seed = int(rng.integers(1 << 31))
    v = _sv(seed)                  # ONE seed makes the vector AND the reset, as
                                   # the collector does: the vector IS the episode
    v["belt_speed"] = C.BELT_SPEED
    # the start pose is part of the episode record (the collector writes
    # current.WAIT_JOINTS into every vector; sample_vector predates it)
    v["wait_joints"] = [float(x) for x in current.WAIT_JOINTS]
    if box is not None:            # held-out box position (boxX condition)
        v["tote_dx"], v["tote_dy"] = float(box[0]), float(box[1])
    face = v["jaw_yaw"]
    env.vec = v
    stop = CS.apply_stop(v)        # same per-episode indexing stop as collection
    CS.apply_tote(v, env)          # ... and the same per-episode box position
    CS.CTX.pop("hook", None)
    r._arm_lo = r.model.actuator_ctrlrange[r._arm_aid, 0]
    r._arm_hi = r.model.actuator_ctrlrange[r._arm_aid, 1]
    if vis:
        visap.apply(vis)
        mount.back = V.DEFAULT["back"] + vis["dback"]
        mount.deg = V.DEFAULT["deg"] + vis["ddeg"]
        # DEFAULT + jitter, not the jitter alone - see the same fix in
        # render_dataset.py.  With DEFAULT["azim"] now 180 (the mount was
        # re-derived after the near-plane fix) assigning the raw deviation put
        # the camera at +-4 deg instead of 180 +-4.
        mount.azim = V.DEFAULT["azim"] + vis["dazim"]
    # the collector's own reset path, verbatim (the same order the bench's
    # load_episode uses): publish the vector, reset, THEN install the start
    # pose.  RandEnv.reset writes HOME, and without install_wait_pose the
    # rollout begins ~1.9 rad away from where every training episode begins.
    CS.CTX.update({k: (x.copy() if isinstance(x, np.ndarray) else x)
                   for k, x in v.items()})
    env.reset(seed=seed)
    CS.install_wait_pose(env)

    idx = next((i for i in range(C.POOL) if r._active[i]), None)
    if idx is None:
        return False, False
    # THE NEW CONTRACT: the policy drives the WHOLE episode from t=0.  Every
    # training episode begins at the wait pose -- arm parked, part at the head
    # of the line -- and the expert does the ~9 s belt wait ITSELF before the
    # chase (the full-episode cache has those frames).  The old handover parked
    # the arm over the stop and waited for a part that a running line never
    # stops for: a state that is not in the data at all.  Measured with a
    # policy that tracks the expert to 0.003 rad on stored frames through wait
    # AND chase: 0/50 pick from the old handover -- the handover state, not the
    # policy, was the failure.
    # History buffer of REAL frames only.  The old code primed it with
    # [f0] * (2*stack+4), i.e. the first decision saw `stack` IDENTICAL frames.
    # No training sample ever looks like that, and probe_prime.py measured what
    # both models do with it: they read "the arm has been static for the whole
    # window" and command the close immediately (s8 -0.772 at entry 0, s3
    # ramping to -0.402 by entry 3), where the same models on consecutive real
    # rows stay flat open (-0.153) and predict the gt close row exactly.
    KEEP = 2 * stack + 4           # stack spans 2*(stack-1)+latency slots
    CAD = CAM_DT                   # capture cadence = policy grid (40 ms)
    hist = []
    rgb_hist = []
    vision = None
    place_vision = None
    if VISUAL_PICK or VISUAL_PLACE:
        from wrist_vision import WristCalibration, PickServo, PlaceServo
        calibration=WristCalibration(r.model,mount)
        visual_dofs=np.array([int(r.model.joint(n).dofadr[0]) for n in
                              ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')])
        if VISUAL_PICK:
            vision = PickServo(r.ik, calibration,VISUAL_CLOSE_S,VISUAL_TAKEOVER_Z)
        if VISUAL_PLACE:
            place_vision=PlaceServo(r.ik,calibration)

    def snap():
        if vision is None and place_vision is None:
            hist.append(current_frame(env, mount, rend))
        else:
            mount.update(r.data)
            rend.update_scene(r.data, camera=mount.cid)
            V.strip_scene_markers(rend)
            rgb = rend.render()
            hist.append(wrist_gray(rgb))
            # implicitfast renders kinematics from BEFORE its final qpos
            # integration. Recover that exact hinge pose from measured q/v;
            # pairing the photo with post-step encoders biases moving views.
            dt=r.model.opt.timestep
            photo_q=r.data.qpos[r._arm_qpos]-dt*r.data.qvel[visual_dofs]
            rgb_hist.append((rgb.copy(),photo_q.copy(),
                             float(r.data.qpos[r._grip_qpos]), r._sim_t-dt))
            del rgb_hist[:max(0,len(rgb_hist)-KEEP)]
        del hist[:max(0, len(hist) - KEEP)]

    t_wait = 0.0
    t_cad = 0.0
    q_hold = r.data.qpos[r._arm_qpos].copy()
    wait5 = q_hold.copy()          # the park reference for ARM_PARK_DB
    # prime the buffer with REAL frames at the wait pose: ~0.8 s of hold, which
    # is exactly how every training episode starts (cache row 0's stack is
    # wait-pose frames padded at the old end).  First decision = the state the
    # data starts from.
    while len(hist) < KEEP:
        E.sim(env, r.model.opt.timestep, arm=q_hold, grip=0.64)
        t_wait += r.model.opt.timestep
        if t_wait - t_cad >= CAD:  # real frames while the arm holds at the wait pose
            t_cad = t_wait
            snap()
            if on_step is not None:
                on_step(dict(phase="wait", t=0.0, hist=hist))
    if on_step is not None:
        on_step(dict(phase="handover", t=0.0, hist=hist))
    # a short belt wait can leave the buffer thin; pad at the OLD end with the
    # earliest real frame instead of fabricating motion
    while len(hist) < KEEP:
        hist.insert(0, hist[0])
    picked = placed = False
    t_pick = t_place = None       # time-to-pick / time-to-place (s) for stats
    action_q = np.zeros((0, 6))
    entry_i = 0                    # absolute 40 ms entry index (TE alignment)
    ens = {}                       # entry index -> [sum of arm votes, weight]
    prev_cmd = None                # last executed arm command (deadband/EMA)
    dwell_until = -1.0             # grasp primitive (see GRASP_DWELL_S)
    hold_arm = None
    prev_shut = False              # commanded-grip edge for the dwell trigger
    entry_dwell = GraspDwell(GRASP_DWELL_S)
    placement = PlacementMonitor(PLACE_SETTLE_S)
    release_ctl = ReleaseOnDescent(release_z=C.TOTE_RELEASE_Z + .06,
                                   lift_z=C.BELT_TOP + .15, lane_y=C.BELT_Y,
                                   retain_on_stall=RETAIN_ON_STALL)
    if trace is not None:
        trace.append(dict(start=True, t=0.0))
    t0 = r._sim_t
    while r._sim_t - t0 < HORIZON_S:
        if len(action_q) == 0:
            # stack: the buffer is on the 40 ms camera grid.  Each training
            # frame shows a state u ~ U(0, 40 ms) STALE (render_dataset); the
            # proprio and the action targets are NOT stale (build_cache anchors
            # both at the nominal obs row).  Reproduce the training staleness
            # exactly by shifting the stack back 0 or 1 slot = 0..40 ms.
            # One slot = one grid step (40 ms).  Each training frame is
            # INDEPENDENTLY u ~ U(0, 40 ms) stale relative to its proprio row
            # (render_dataset draws one latency per frame), so stack slot k is
            # k*40 ms + lat_k old with lat_k iid - reproduce that by drawing
            # each slot's 0/1-slot shift independently (Act 19: the old code
            # drew ONE shift for the whole stack, a coherent shift that is not
            # the training distribution).  NB: not `idx` - that name holds the
            # part index used below.
            sel = [-1 - k - int(lrng.integers(0, 2)) for k in range(stack)]
            if vision is not None or place_vision is not None:
                rgb,vq,vg,vt=rgb_hist[sel[0]]
                if vision is not None:
                    vision.observe(rgb,vq,vg,vt-t0)
                if place_vision is not None:
                    place_vision.observe(rgb,vq,vg,vt-t0,release_ctl.lifted,
                                         release_ctl.released_at is not None)
            img = torch.stack([hist[i] for i in sel])[None].to(dev)
            # proprio = IST || SOLL, built in the SAME order as dataset.py.
            # SOLL is `data.ctrl` read directly: expert.sim and this loop both
            # write their command there and hold it for the step, so ctrl IS
            # the currently commanded target.  The width is the model's own
            # (ckpt 'qdim', default 6) so pre-SOLL checkpoints still evaluate.
            ist = np.append(r.data.qpos[r._arm_qpos],
                            r.data.qpos[r._grip_qpos])
            if getattr(model, "qdim", 6) > len(ist):
                soll = np.append(r.data.ctrl[r._arm_aid],
                                 r.data.ctrl[r._grip_aid])
                q6 = np.concatenate([ist, soll])
            else:
                q6 = ist
            q = torch.from_numpy(q6.astype(np.float32))[None].to(dev)
            with torch.no_grad():
                if CARRY_HEAD is not None and release_ctl.lifted and release_ctl.released_at is None:
                    from carry_policy import carry_features
                    features = carry_features(model, img, q)
                    base = model.head(features).reshape(-1, CHUNK, 6)
                    base[..., :5] = CARRY_HEAD(features)[..., :5]
                    act = base[0].cpu().numpy()
                else:
                    act = model(img, q)[0].cpu().numpy()
            act_raw = act
            act = np.clip(act, -1, 1)
            if TE_M > 0.0:
                # ACT temporal ensembling (arm only).  Every overlapping chunk
                # votes on the entry it predicted for this absolute index;
                # weights decay one factor per decision, so the newest plan
                # dominates and older plans only smooth the seam.  The grip
                # channel is NOT averaged: its timing must be the newest plan's.
                dec = float(np.exp(-TE_M))
                action_q = temporal_actions(ens, act, entry_i, EXEC, dec)
            else:
                action_q = act[:EXEC]     # execute EXEC, re-observe
            if CARRY_LOOKAHEAD and release_ctl.lifted and release_ctl.released_at is None:
                action_q = carry_lookahead(act, EXEC, CARRY_LOOKAHEAD)
            if GRIP_SNAP:
                action_q[:, 5] = np.where(action_q[:, 5] < GRIP_THRESH,
                                          GRIP_SHUT_N, GRIP_OPEN_N)
            shut_now = bool(action_q[0, 5] < GRIP_THRESH)
            if DWELL_MODE == "decision" and shut_now and not prev_shut and GRASP_DWELL_S > 0:
                # the close COMMAND, on the open->shut edge only: hold_arm
                # already holds the command in force right now, which is what
                # the arm must hold while the jaws shut (see GRASP_DWELL_S)
                dwell_until = r._sim_t + GRASP_DWELL_S
            prev_shut = shut_now
            if GRIP_GATE_R is not None:
                near = np.linalg.norm(
                    r._cube_pos(idx)[:2]
                    - r.data.site_xpos[r._grasp_site][:2]) < GRIP_GATE_R
                if not near:
                    action_q[:, 5] = GRIP_OPEN_N
                    dwell_until = -1.0
                    prev_shut = False   # the forced open breaks the edge too
            if on_step is not None:
                on_step(dict(phase="decision", t=round(r._sim_t - t0, 4),
                             hist=hist, act=act_raw.copy(),
                             policy_image=img, carry_lifted=release_ctl.lifted,
                             primitive_release_t=release_ctl.released_at,
                             raw_grip=act_raw[:, 5].copy(),
                             commanded=action_q[:, 5].copy(), q6=q6.copy()))
            if trace is not None:
                # gt grip is a step: open -0.151 -> shut -0.797 (normalised),
                # so the first entry below -0.4 is the close command.  The
                # expert closes ~1.0-1.2 s after cache row 0, and the cache
                # window starts 0.155 s after this handover (provenance).
                trace.append(dict(t=round(r._sim_t - t0, 4),
                                  raw_grip=[round(float(x), 4) for x in act_raw[:, 5]],
                                  grip_cmd=[round(float(x), 3) for x in act[:, 5]],
                                  gq=round(float(r.data.qpos[r._grip_qpos]), 4),
                                  site=[round(float(x), 4) for x in
                                        r.data.site_xpos[r._grasp_site]],
                                  part=[round(float(x), 4) for x in
                                        r._cube_pos(idx)],
                                  gap_mm=round(float(np.linalg.norm(
                                      r._jaws()[0] - r._jaws()[1])) * 1000, 2),
                                  cmd=[round(float(x), 4) for x in action_q[0][:5]],
                                  ist=[round(float(x), 4) for x in
                                       r.data.qpos[r._arm_qpos]],
                                  dwell=round(float(r._sim_t - dwell_until), 3),
                                  picked_p=bool(r._grasped() == idx),
                                  release_confirmed=release_ctl.confirmed,
                                  release_lifted=release_ctl.lifted,
                                  primitive_release_t=release_ctl.released_at))
        a = action_q[0]
        action_q = action_q[1:]
        arm = r._mid + r._half * a[:5]
        if ARM_MAX_DQ > 0:           # keep the fed-back SOLL on the expert's
            ist5 = r.data.qpos[r._arm_qpos]   # steady-chase envelope
            arm = np.clip(arm, ist5 - ARM_MAX_DQ, ist5 + ARM_MAX_DQ)
        # shiver controls: applied BEFORE the dwell/pin bookkeeping so the
        # frozen grasp pose stores the command that is actually executed
        db = ARM_DEADBAND
        if ARM_PARK_R > 0.0 and float(np.max(np.abs(arm - wait5))) < ARM_PARK_R:
            db = max(db, ARM_PARK_DB)
        if db > 0.0 and prev_cmd is not None:
            d = arm - prev_cmd
            arm = np.where(np.abs(d) < db, prev_cmd, arm)
        if ARM_EMA > 0.0 and prev_cmd is not None:
            arm = (1.0 - ARM_EMA) * prev_cmd + ARM_EMA * arm
        prev_cmd = arm.copy()
        if ARM_PIN_S and hold_arm is not None and r._sim_t - t0 < ARM_PIN_S                 and r._sim_t >= dwell_until:
            arm = hold_arm
        if DWELL_MODE == "decision" and r._sim_t < dwell_until:
            arm = hold_arm if hold_arm is not None else arm
        elif DWELL_MODE == "decision":
            hold_arm = arm
        if DWELL_MODE == "entry":
            arm = entry_dwell.step(r._sim_t, arm, bool(a[5] < GRIP_THRESH))
            dwell_until = entry_dwell.until
            hold_arm = arm.copy()
        grip = r._grip_lo + (a[5] + 1.0) * 0.5 * (r._grip_hi - r._grip_lo)
        visual_shut = None
        if vision is not None:
            correction = vision.step(r._sim_t-t0,r.data.qpos[r._arm_qpos].copy(),
                                     float(r.data.qpos[r._grip_qpos]))
            if correction is not None:
                arm,visual_shut=correction
                grip_n=GRIP_SHUT_N if visual_shut else GRIP_OPEN_N
                grip=r._grip_lo+(grip_n+1)*.5*(r._grip_hi-r._grip_lo)
        if place_vision is not None:
            correction=place_vision.step(r._sim_t-t0,r.data.qpos[r._arm_qpos].copy(),
                                         float(r.data.qpos[r._grip_qpos]),arm,
                                         release_ctl.lifted,release_ctl.released_at is not None)
            if correction is not None:
                arm,visual_shut=correction
                grip_n=GRIP_SHUT_N if visual_shut else GRIP_OPEN_N
                grip=r._grip_lo+(grip_n+1)*.5*(r._grip_hi-r._grip_lo)
        if RELEASE_PRIMITIVE:
            policy_shut = bool(a[5] < GRIP_THRESH) if visual_shut is None else visual_shut
            arm, shut = release_ctl.step(
                r._sim_t - t0, arm, float(r.data.qpos[r._grip_qpos]),
                policy_shut, r.data.site_xpos[r._grasp_site])
            # Preserve the original numerical command until an override is
            # needed; even tiny float32/float64 differences can change contact.
            if shut != policy_shut:
                grip_n = GRIP_SHUT_N if shut else GRIP_OPEN_N
                grip = r._grip_lo + (grip_n + 1) * .5 * (r._grip_hi - r._grip_lo)
        # chunk entries sit on the 25 Hz policy grid (CAM_DT/DT = 8 physics
        # steps): hold each for exactly 8 steps, else the policy would run at
        # a different rate than it was trained on (train/eval mismatch)
        if trace is not None:
            # Legacy 'cmd' is the base policy suggestion. Preserve it, and
            # explicitly record the targets actually sent after all overrides.
            trace[-1].setdefault('executed',[]).append(dict(
                t=round(r._sim_t-t0,4),arm=np.asarray(arm).tolist(),grip=float(grip),
                visual_pick_phase=vision.phase if vision is not None else None,
                visual_place_phase=place_vision.phase if place_vision is not None else None))
        E.sim(env, ENTRY_STEPS * r.model.opt.timestep, arm=arm, grip=float(grip))
        # capture a frame every entry boundary (40 ms): at the next decision
        # the stack must be one grid step apart like training, newest first
        snap()
        entry_i += 1
        if r._grasped() == idx:
            if not picked:       # first entry, not every later one
                picked = True
                t_pick = r._sim_t - t0
        if picked and placement.step(
                r._sim_t - t0, r._in_tote(r._cube_pos(idx)), r._grasped() == idx,
                float(np.linalg.norm(r.data.qvel[r._cube_dof[idx]:r._cube_dof[idx] + 3]))):
            placed = True
            t_place = r._sim_t - t0
        if on_step is not None:
            on_step(dict(phase="policy", t=round(r._sim_t - t0, 4), hist=hist,
                         act=a, arm=arm, grip=float(grip),
                         site=r.data.site_xpos[r._grasp_site].copy(),
                         part=r._cube_pos(idx).copy(),
                         gap_mm=float(np.linalg.norm(
                             r._jaws()[0] - r._jaws()[1])) * 1000,
                         grasped=bool(r._grasped() == idx),
                         picked=bool(picked), placed=bool(placed),
                         q6=np.append(r.data.qpos[r._arm_qpos],
                                      r.data.qpos[r._grip_qpos]).copy()))
        if placed:
            break  # publish the successful final step to the live viewer first
    if vis:
        visap.restore()
        mount.back = V.DEFAULT["back"]
        mount.deg = V.DEFAULT["deg"]
        mount.azim = V.DEFAULT["azim"]
    if stats is not None:         # timing telemetry (watch.py passes nothing)
        stats.update(picked=bool(picked), placed=bool(placed),
                     seed=seed, tote_dx=v["tote_dx"], tote_dy=v["tote_dy"],
                     held_at_end=bool(r._grasped() == idx),
                     tote_enter_t=placement.entered_at,
                     primitive_release_t=release_ctl.released_at,
                     place_criterion=("released-settled" if PLACE_SETTLE_S > 0 else "tote-entry"),
                     pick_t=t_pick,
                     place_t=(t_place if placed else None),
                     end_t=round(r._sim_t - t0, 2),
                     horizon_hit=not placed)
        if vision is not None:
            stats['visual_pick_events']=vision.events
        if place_vision is not None:
            stats['visual_place_events']=place_vision.events
    return picked, placed


def main():
    global GRIP_SNAP, EXEC, GRIP_GATE_R, GRASP_DWELL_S, GRIP_THRESH
    global TE_M, ARM_MAX_DQ, ARM_DEADBAND, ARM_EMA, ARM_PARK_R, ARM_PARK_DB
    global DWELL_MODE
    global PLACE_SETTLE_S
    global RELEASE_PRIMITIVE
    global RETAIN_ON_STALL
    global CARRY_LOOKAHEAD
    global CARRY_HEAD
    global VISUAL_PICK
    global VISUAL_PLACE
    global VISUAL_CLOSE_S, VISUAL_TAKEOVER_Z
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=current.CKPT)
    ap.add_argument("--ds", default=current.DS,
                    help="dataset whose stored line record configures the "
                         "belt: without it the evaluator runs the OLD "
                         "stopping line and judges the policy in a world it "
                         "was never trained in")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument('--start-index',type=int,default=0,
                    help='skip completed episodes in this exact seed stream (repeat=1 only)')
    ap.add_argument("--trace", default=None,
                    help="write per-decision grip traces to this json file")
    ap.add_argument("--out", default=None, help="explicit evaluation report path")
    ap.add_argument("--dwell-mode", choices=("decision", "entry"), default=DWELL_MODE,
                    help="legacy decision-edge hold or every executed grip edge")
    ap.add_argument("--place-settle-s", type=float, default=PLACE_SETTLE_S,
                    help="evaluation only: require release, tote retention and "
                         "speed <= 0.05 m/s for this duration (0 = legacy tote entry)")
    ap.add_argument("--release-primitive", action=argparse.BooleanOptionalAction,
                    default=RELEASE_PRIMITIVE,
                    help="release after a confirmed jaw stall, "
                         "lift, and low descent away from belt; robot proprio/FK only")
    ap.add_argument("--retain-on-stall", action="store_true",
                    help="experimental: suppress premature opens after a stable "
                         "jaw stall, including before lift (requires release primitive)")
    ap.add_argument("--carry-lookahead", type=int, default=0,
                    help="experimental: after a confirmed lift, use arm targets "
                         "N chunk entries ahead, retaining newest grip timing")
    ap.add_argument("--carry-ckpt", default=None,
                    help="experimental carry-only head, enabled after confirmed lift")
    ap.add_argument('--visual-pick',action=argparse.BooleanOptionalAction,default=VISUAL_PICK,
                    help='experimental calibrated RGB wrist servo for approach/grasp/lift')
    ap.add_argument('--visual-place',action=argparse.BooleanOptionalAction,default=VISUAL_PLACE,
                    help='experimental RGB tote-silhouette alignment after grasp/lift')
    ap.add_argument('--visual-close-s',type=float,default=VISUAL_CLOSE_S)
    ap.add_argument('--visual-takeover-z',type=float,default=VISUAL_TAKEOVER_Z)
    ap.add_argument("--no-snap", action="store_true",
                    help="disable the bistable grip decode (ablation)")
    ap.add_argument("--exec", type=int, default=EXEC,
                    help="chunk entries executed before re-observing")
    ap.add_argument("--grip-gate-r", type=float, default=None,
                    help="DIAGNOSTIC: force the grip open while the part is "
                         "farther than this many metres from the grasp site")
    ap.add_argument("--dwell", type=float, default=GRASP_DWELL_S,
                    help="grasp dwell (s): freeze the arm while the jaws shut, "
                         "once per close command (0 = off)")
    ap.add_argument("--grip-thresh", type=float, default=GRIP_THRESH,
                    help="bistable grip decode threshold (normalised; the "
                         "label modes are -0.1515/-0.797, midpoint -0.474)")
    ap.add_argument("--conds", default="nominal,tierB,boxX",
                    help="comma-separated subset of nominal,tierB,boxX")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each condition this many times over fresh "
                         "episode seeds and report the median across reps "
                         "(a single pass is a coin-flip sample)")
    ap.add_argument("--seed-base", type=int, default=4242,
                    help="episode-seed stream base (4242 = the development "
                         "set every ckpt so far was selected on). Use a "
                         "different value (e.g. 9999) for a FRESH-seed "
                         "confirmation unseen during model selection.")
    ap.add_argument("--te", type=float, default=TE_M,
                    help="ACT temporal ensembling decay per decision (0 = "
                         "off; 0.1 = older chunks vote exp(-0.1*age)). "
                         "Smooths the 80 ms chunk seam the arm shivers on.")
    ap.add_argument("--arm-max-dq", type=float, default=None,
                    help="rate-limit the arm command to N rad per 40 ms entry "
                         "relative to the current joint state (0 = off; "
                         "expert steady-chase p95 = 0.103 rad)")
    ap.add_argument("--arm-deadband", type=float, default=None,
                    help="latch the arm command while it stays within N rad of "
                         "the previous entry (0 = off; parked tremor is "
                         "0.008-0.011 rad, real motion 0.05-0.13)")
    ap.add_argument("--arm-ema", type=float, default=None,
                    help="first-order arm smoothing weight on the new command "
                         "(0.5 = 40 ms lag; 0 = off)")
    ap.add_argument("--arm-park-r", type=float, default=None,
                    help="radius (rad) around the wait pose that counts as "
                         "parked for --arm-park-db (0 = off)")
    ap.add_argument("--arm-park-db", type=float, default=None,
                    help="strong arm latch used while parked (0.02 kills the "
                         "waited-arm buzz; default 0.02)")
    args = ap.parse_args()
    ramcap.install(threads=1)
    if args.n <= 0 or args.repeat <= 0 or not 1 <= args.exec <= CHUNK:
        ap.error("n/repeat must be positive and exec must be in 1..8")
    if args.start_index<0 or (args.start_index and args.repeat!=1):
        ap.error('start-index must be nonnegative and requires repeat=1')
    if args.visual_place and not args.release_primitive:
        ap.error('visual-place requires release-primitive for proprioceptive grasp/lift confirmation')
    if args.visual_pick and (not 0<args.visual_close_s<=2 or not .64<=args.visual_takeover_z<=.9):
        ap.error('visual-pick requires close-s in (0,2] and takeover-z in [.64,.9]')
    if (args.visual_pick or args.visual_place) and args.grip_gate_r is not None:
        ap.error('privileged grip-gate diagnostics must disable visual-pick and visual-place')
    conds = {c.strip() for c in args.conds.split(",") if c.strip()}
    if not conds or not conds <= {"nominal", "tierB", "boxX"}:
        ap.error("conds must be a nonempty subset of nominal,tierB,boxX")
    if args.dwell < 0 or args.place_settle_s < 0:
        ap.error("dwell and place-settle-s must be nonnegative")
    if args.retain_on_stall and not args.release_primitive:
        ap.error("retain-on-stall requires release-primitive")
    if not 0 <= args.carry_lookahead < CHUNK:
        ap.error("carry-lookahead must be in 0..7")
    if args.carry_lookahead and (not args.release_primitive or args.te):
        ap.error("carry-lookahead requires release-primitive and te=0")
    if args.carry_ckpt and (not args.release_primitive or args.carry_lookahead or args.te):
        ap.error("carry-ckpt requires release-primitive, carry-lookahead=0 and te=0")
    out = args.out or (_os.path.splitext(args.trace)[0] + ".times.json" if args.trace else
                      f"eval_times_{_os.path.basename(args.ckpt).replace('.pt', '')}.json")
    for path in (out, args.trace):
        if path:
            _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    def sha256(path):
        with open(path, "rb") as src:
            return hashlib.file_digest(src, "sha256").hexdigest()
    _here = _os.path.dirname(_os.path.abspath(__file__))
    def _src(name):                      # modules live next to this file
        return _os.path.join(_here, name)
    run_hashes = {p: sha256(p) for p in
                  (args.ckpt, __file__, _src("policy_runtime.py"), _src("model.py"),
                   _src("collect_state.py"), _src("build_cell.py"), _src("conveyor_env.py"), _src("current.py"),
                   _src("arm_ik.py"), _src("cam_variant.py"), _src("render_dataset.py"), _src("ramcap.py"))}
    if args.carry_ckpt:
        run_hashes.update({p: sha256(p) for p in (args.carry_ckpt, _src("carry_policy.py"))})
    if args.visual_pick or args.visual_place:
        run_hashes[_src('wrist_vision.py')]=sha256(_src('wrist_vision.py'))
    # Each run has its own journal, available even if interrupted before the
    # final aggregate. Never confuse a partial run with a completed report.
    journal = out + ".episodes.jsonl"
    with open(journal, "w") as f:
        f.write(json.dumps(dict(event="start", args=vars(args))) + "\n")
    DWELL_MODE = args.dwell_mode
    PLACE_SETTLE_S = args.place_settle_s
    RELEASE_PRIMITIVE = args.release_primitive
    RETAIN_ON_STALL = args.retain_on_stall
    CARRY_LOOKAHEAD = args.carry_lookahead
    VISUAL_PICK = args.visual_pick
    VISUAL_PLACE = args.visual_place
    VISUAL_CLOSE_S = args.visual_close_s
    VISUAL_TAKEOVER_Z = args.visual_takeover_z
    GRIP_SNAP = not args.no_snap
    EXEC = args.exec
    GRIP_GATE_R = args.grip_gate_r
    GRASP_DWELL_S = float(args.dwell)
    GRIP_THRESH = float(args.grip_thresh)
    TE_M = float(args.te)
    if args.arm_max_dq is not None:
        ARM_MAX_DQ = float(args.arm_max_dq)
    if args.arm_deadband is not None:
        ARM_DEADBAND = float(args.arm_deadband)
    if args.arm_ema is not None:
        ARM_EMA = float(args.arm_ema)
    if args.arm_park_r is not None:
        ARM_PARK_R = float(args.arm_park_r)
    if args.arm_park_db is not None:
        ARM_PARK_DB = float(args.arm_park_db)

    import gpu_guard
    dev = gpu_guard.pick_cuda()
    ck = torch.load(args.ckpt, weights_only=True, map_location=dev)
    stack = int(ck.get("stack", STACK))
    print(f"history: {stack} frames = {(stack - 1) * 40} ms lookback at 25 Hz"
          f"  exec {EXEC} x 40 ms = {EXEC * 40} ms open loop")
    qdim = int(ck.get("qdim", 6))
    model = Policy(img_ch=stack, qdim=qdim).to(dev)
    load_state(model, ck["model"], args.ckpt)
    model.eval()
    CARRY_HEAD = None
    if args.carry_ckpt:
        from carry_policy import CarryHead
        carry_ck = torch.load(args.carry_ckpt, weights_only=True, map_location=dev)
        if carry_ck["base_sha256"] != run_hashes[args.ckpt]:
            raise ValueError("carry head was trained with a different base encoder checkpoint")
        CARRY_HEAD = CarryHead(model).to(dev).eval()
        CARRY_HEAD.load_state_dict(carry_ck["model"], strict=True)
    print(f"loaded {args.ckpt} (epoch {ck['epoch']}, val {ck['val']:.5f})")
    print(f"proprio: {qdim} = IST 6" + (" + SOLL 6 (gripper tracking error "
          "is a direct held-object signal)" if qdim > 6 else " (no SOLL)"))

    env = RandEnv(ConveyorPickEnv())
    # same trap the render had: without the line record the belt stops at the
    # indexing stop, so a policy recorded on a running line is evaluated in a
    # world where the part is somewhere else entirely
    import json as _json
    _rows = [l for l in open(f"{args.ds}/manifest.jsonl") if l.strip()]
    if _rows:
        CS.apply_line_config(_json.loads(_rows[0]).get("line"))
    print("line:", CS.RUNNING_LINE and "RUNNING" or "stopping",
          C.BELT_SPEED, "m/s")
    visap = VisApply(env.real.model)
    mount = V.Mount(env.real.model)
    rend = mujoco.Renderer(env.real.model, height=128, width=128)
    rend._scene_option.geomgroup[:] = [1] * 6
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    results = {}
    all_traces = []
    all_eps = []                 # per-episode timing records (pick_t/place_t)
    conds = {c.strip() for c in args.conds.split(",") if c.strip()}
    for name, want_vis, box in (("nominal", False, None), ("tierB", True, None),
                                ("boxX", False, "corner")):
        if name not in conds:
            continue
        rng = np.random.default_rng(args.seed_base + (1 if want_vis else 0))
        for _ in range(args.start_index):
            if want_vis:
                sample_vis(rng)
            rng.integers(1<<31)
        # --repeat K: K passes over K fresh-seed episode sets.  A single pass
        # is a coin-flip sample (n=8 bins are 12.5% wide); the median over K
        # INDEPENDENT episode sets is the firmer number, no bootstrap needed.
        R = max(1, args.repeat)
        rep_rows = []
        eps_cond = []
        t0 = time.perf_counter()
        for rep in range(R):
            picks, places = [], []
            for i in range(args.start_index,args.start_index+args.n):
                vis = sample_vis(rng) if want_vis else None
                # boxX keeps walking the corner sequence ACROSS reps (corner
                # count 3 < n), so rep 0 is bit-identical to the old single
                # pass and later reps extend the walk instead of restarting it
                box_off = (BOX_X_CORNERS[(rep * args.n + i) % len(BOX_X_CORNERS)]
                           if box else None)
                tr = [] if args.trace else None
                # the latency draw gets its OWN stream: otherwise the episode
                # sequence (and therefore which episodes a run sees) changes
                # with EXEC/STACK, which makes ablations incomparable
                st = {}
                picked, placed = rollout(env, model, mount, rend, visap, vis,
                                         rng, stack, tr,
                                         lrng=np.random.default_rng(
                                             777 + rep * 1000 + i),
                                         box=box_off, stats=st)
                if tr is not None:
                    all_traces.append(dict(cond=name, rep=rep, i=i,
                                           picked=bool(picked),
                                           placed=bool(placed), trace=tr))
                st.update(cond=name, rep=rep, i=i)
                with open(journal, "a") as f:
                    f.write(json.dumps(st) + "\n")
                eps_cond.append(st)
                picks.append(picked)
                places.append(placed)
                print(f"  [{name} r{rep} {i-args.start_index+1:2d}/{args.n}] pick "
                      f"{int(picked)} place {int(placed)}", flush=True)
            rep_rows.append((float(np.mean(picks)), float(np.mean(places))))
        pk = np.array([r[0] for r in rep_rows])
        pl = np.array([r[1] for r in rep_rows])
        results[name] = dict(
            pick=float(pk.mean()), place=float(pl.mean()),
            pick_per_rep=[round(float(x), 4) for x in pk],
            place_per_rep=[round(float(x), 4) for x in pl],
            pick_med=float(np.median(pk)), place_med=float(np.median(pl)),
            wall_s=round(time.perf_counter() - t0, 1))
        all_eps.extend(eps_cond)
        # time-to-pick vs the 20 s horizon measures WHERE the place-rate dies:
        # a pick that lands near the horizon cannot be carried and dropped in
        # time -- that is the measured answer to "grasps but never places".
        pk_t = [e["pick_t"] for e in eps_cond
                if e["picked"] and e.get("pick_t") is not None]
        pl_t = [e["place_t"] for e in eps_cond
                if e["placed"] and e.get("place_t") is not None]
        late = sum(1 for e in eps_cond if e["picked"] and not e["placed"]
                   and e.get("end_t", 0.0) >= HORIZON_S - 0.1)
        msg = (f"{name}: pick {pk.mean()*100:.1f}% (med {np.median(pk)*100:.1f}%)  "
               f"place {pl.mean()*100:.1f}% (med {np.median(pl)*100:.1f}%)")
        if pk_t:
            msg += f"  t_pick med {np.median(pk_t):.1f}s max {max(pk_t):.1f}s"
        if pl_t:
            msg += f"  t_place med {np.median(pl_t):.1f}s"
        if late:
            msg += f"  picked-not-placed-at-horizon: {late}"
        print(msg + "\n")

    print(json.dumps(results, indent=2))
    if args.trace:
        with open(args.trace, "w") as f:
            json.dump(all_traces, f)
        print(f"traces -> {args.trace}")
    with open(out, "w") as f:
        json.dump(dict(ckpt=args.ckpt, n=args.n, repeat=args.repeat,
                       args=vars(args), seed_base=args.seed_base,
                       protocol=("v3-released-settled" if PLACE_SETTLE_S > 0 else
                                 "v2-entry-dwell" if DWELL_MODE == "entry" else "v2"),
                       trace_command_schema='base suggestion in cmd; actual raw targets in executed',
                       runtime=dict(exec=EXEC, dwell=GRASP_DWELL_S,
                                    dwell_mode=DWELL_MODE, te=TE_M,
                                    place_settle_s=PLACE_SETTLE_S,
                                    release_primitive=RELEASE_PRIMITIVE,
                                    retain_on_stall=RETAIN_ON_STALL,
                                    carry_lookahead=CARRY_LOOKAHEAD,
                                    carry_ckpt=args.carry_ckpt,
                                    visual_pick=VISUAL_PICK,visual_place=VISUAL_PLACE,
                                    visual_close_s=VISUAL_CLOSE_S,
                                    visual_takeover_z=VISUAL_TAKEOVER_Z,
                                    ram_cap_gb=ramcap.CAP_GB,
                                    hard_ram_cap=ramcap._capped is not None,
                                    grip_snap=GRIP_SNAP, grip_thresh=GRIP_THRESH,
                                    arm_max_dq=ARM_MAX_DQ, arm_ema=ARM_EMA,
                                    arm_deadband=ARM_DEADBAND,
                                    arm_park_r=ARM_PARK_R, arm_park_db=ARM_PARK_DB),
                       hashes=run_hashes,
                       results=results, episodes=all_eps), f)
    print(f"per-episode timing -> {out}")
    env.real.close()
    rend.close()


if __name__ == "__main__":
    main()
