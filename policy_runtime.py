"""Small, observation-only helpers shared by policy evaluation and the viewer."""
import numpy as np
import torch


def temporal_actions(votes, act, entry, execute, decay):
    """Blend aligned arm predictions; grip always follows the newest chunk."""
    for old in [k for k in votes if k < entry]:
        del votes[old]
    for cell in votes.values():
        cell[0] *= decay
        cell[1] *= decay
    for k in range(len(act)):
        cell = votes.setdefault(entry + k, [np.zeros(5), 0.0])
        cell[0] += act[k, :5]
        cell[1] += 1.0
    out = np.zeros((execute, act.shape[1]))
    for j in range(execute):
        total, weight = votes[entry + j]
        out[j, :5] = total / weight
        out[j, 5] = act[j, 5]
    return out


def carry_lookahead(act, execute, offset):
    """Use the policy's later arm targets while preserving newest grip timing."""
    out = act[:execute].copy()
    indices = np.minimum(np.arange(execute) + offset, len(act) - 1)
    out[:, :5] = act[indices, :5]
    return out


def wrist_gray(rgb):
    """Match torch area(128 -> 96), RGB mean and uint8 quantisation exactly.

    For this ratio each adaptive-area window is 2x2. Summing uint8 samples
    before conversion avoids launching a CPU thread pool for every camera
    frame. No renderer flags, pixels, cadence or latency draws change.
    """
    rgb = np.asarray(rgb)
    if rgb.shape != (128, 128, 3) or rgb.dtype != np.uint8:
        raise ValueError("expected a 128x128 RGB uint8 wrist frame")
    starts = np.arange(96) * 128 // 96
    sums = np.zeros((96, 96), dtype=np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            sums += rgb[starts[:, None] + dy, starts[None, :] + dx].sum(
                axis=2, dtype=np.float32)
    gray = np.rint(sums / np.float32(12)).clip(0, 255)
    return torch.from_numpy(gray) / 255.0


class GraspDwell:
    """Detect executed grip edges, including inside an open-loop chunk.

    Holds the preceding arm command just as the demonstration does. An open
    command cancels the hold; a sustained close never re-triggers it.
    """
    def __init__(self, seconds):
        self.seconds = float(seconds)
        self.shut = False
        self.until = -1.0
        self.arm = None

    def step(self, now, arm, shut):
        if shut and not self.shut:
            self.until = now + self.seconds
        elif not shut:
            self.until = -1.0
        self.shut = bool(shut)
        if now < self.until and self.arm is not None:
            return self.arm.copy()
        self.arm = np.asarray(arm).copy()
        return self.arm.copy()


class PlacementMonitor:
    """Evaluation only: distinguish tote entry from released, settled placement."""
    def __init__(self, settle_s=0.0, max_speed=0.05):
        self.settle_s = float(settle_s)
        self.max_speed = float(max_speed)
        self.since = None
        self.entered_at = None

    def step(self, now, in_tote, held, speed):
        if in_tote and self.entered_at is None:
            self.entered_at = float(now)
        if self.settle_s == 0:
            return bool(in_tote)  # historical metric, for matched comparisons
        if not in_tote or held or speed > self.max_speed:
            self.since = None
            return False
        if self.since is None:
            self.since = float(now)
        return now - self.since >= self.settle_s - 1e-9


class ReleaseOnDescent:
    """Optional hybrid release using robot proprioception/FK only.

    Confirm a stalled closed jaw, observe a lift, then release on a low
    descent away from the conveyor. The fixture height is configuration;
    no cube pose, tote offset, simulator contact flag or scoring flag enters
    this controller. The vision policy still supplies every arm target.
    """
    def __init__(self, release_z=.735, lift_z=.775, lane_y=.245,
                 retain_on_stall=False):
        self.release_z = release_z
        self.lift_z = lift_z
        self.lane_y = lane_y
        self.retain_on_stall = bool(retain_on_stall)
        self.last_gq = None
        self.stalled_since = None
        self.confirmed = False
        self.lifted = False
        self.released_at = None
        self.hold_arm = None

    def step(self, now, arm, gq, commanded_shut, site):
        arm = np.asarray(arm).copy()
        if self.released_at is not None:
            if now - self.released_at < .8:
                arm = self.hold_arm.copy()
            return arm, False
        stable = self.last_gq is not None and abs(gq - self.last_gq) < .012
        self.last_gq = float(gq)
        # Open jaws (.64) and fully empty jaws (-.1745) cannot confirm a grasp.
        stalled = commanded_shut and -.04 < gq < .4 and stable
        if stalled:
            if self.stalled_since is None:
                self.stalled_since = float(now)
            if now - self.stalled_since >= .16 - 1e-9:
                self.confirmed = True
        else:
            self.stalled_since = None
        if gq < -.10:  # empty close: don't turn a missed pick into a carry latch
            self.confirmed = self.lifted = False
        if self.confirmed and site[2] >= self.lift_z:
            self.lifted = True
        if self.lifted and site[1] < self.lane_y - .12 and site[2] <= self.release_z:
            self.released_at = float(now)
            self.hold_arm = arm.copy()
            return arm, False
        retain = self.lifted or (self.retain_on_stall and self.confirmed)
        return arm, True if retain else bool(commanded_shut)
