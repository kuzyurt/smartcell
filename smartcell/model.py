"""Phase 4 model: ~1M-param wrist-camera BC policy with action chunking.

Input  N x 96 x 96 stacked wrist frames (uint8/255) + proprio (12), N = history
       length (25 Hz grid: N frames at 40 ms -> (N-1)*40 ms of lookback)
       proprio = IST(6) || SOLL(6), see QPOS below
Output 8-step action chunk @25 Hz  ->  (8, 6) in [-1, 1]

The chunk is the standard BC trick: the policy predicts 8 future actions from
one observation and executes EXEC=2 of them before re-observing (80 ms open
loop, evaluate.py), which smooths control at the camera's own rate.
"""
import numpy as np
import torch
import torch.nn as nn

print("torch", torch.__version__)

IMG_CH, IMG_SZ, CHUNK, ACT = 3, 96, 8, 6
# Proprio width = IST(6) || SOLL(6).
#
#   IST   the measured joint state, qpos: 5 arm joints + the gripper jaw.
#   SOLL  the actuator targets actually COMMANDED at that instant, i.e.
#         data.ctrl at eval and `act_raw` in the collected episodes.
#
# They are in the same units (radians, straight from the model's ctrlrange), so
# SOLL - IST is the tracking error and the network can read it off with one
# subtraction.  For the gripper that error is a causal "is something between my
# jaws" sensor and needs no vision at all: measured on the expert (see
# probe_soll_ist.py) a settled close stalls at |SOLL-IST| = 0.18829 with the
# 32 mm part held, against 0.00000-0.00152 with the jaws empty - disjoint, a
# 99.2% dead zone, and 14.8 mm of blocked jaw travel.
#
# Until this change the policy got IST only, and the image - which in every
# dataset up to ds_0004 did not even contain the gripper, because of the
# renderer near-plane defect fixed in cam_variant.apply_clip_fix - was the only
# thing that could hint at whether a part was already held.
QPOS = 12


def _block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Policy(nn.Module):
    """DrQ-style small convnet; depthwise-separable head keeps params ~1M."""

    def __init__(self, img_ch=IMG_CH, qdim=QPOS):
        super().__init__()
        # qdim is a parameter so the pre-SOLL 6-wide checkpoints still load:
        # evaluate.py/watch.py read it back from the ckpt (default 6).
        self.qdim = int(qdim)
        self.in_ch = int(img_ch)            # history length == image channels
        self.stem = nn.Sequential(          # 96 -> 48
            nn.Conv2d(img_ch, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.b1 = _block(48, 72, 2)         # 48 -> 24
        self.b2 = _block(72, 128, 2)        # 24 -> 12
        self.b3 = _block(128, 192, 2)       # 12 -> 6
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.prop = nn.Sequential(nn.Linear(self.qdim, 128),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(192 + 128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, CHUNK * ACT), nn.Tanh())

    def forward(self, img, qpos):
        f = self.pool(self.b3(self.b2(self.b1(self.stem(img))))).flatten(1)
        p = self.prop(qpos)
        return self.head(torch.cat([f, p], dim=1)).view(-1, CHUNK, ACT)


# Keys that old checkpoints carry but the model no longer defines.  The old
# `c1` branch (10,208 params) was registered and saved but forward() never
# called it; it is removed, and load_state strips its keys EXPLICITLY so a
# legacy checkpoint still loads strict for everything else.
LEGACY_KEYS = ("c1.",)


def load_state(model, state, path=""):
    """load_state_dict with explicit legacy-key stripping (strict otherwise).

    Returns the stripped key count; a mismatch that is NOT a known legacy key
    still raises, so a wrong architecture cannot load silently.
    """
    drop = [k for k in list(state) if k.startswith(LEGACY_KEYS)]
    for k in drop:
        del state[k]
    if drop:
        print(f"load_state: stripped {len(drop)} legacy key(s) "
              f"({', '.join(sorted({k.split('.')[0] for k in drop}))}) "
              f"from {path or 'checkpoint'}")
    model.load_state_dict(state, strict=True)
    return len(drop)


if __name__ == "__main__":
    ch = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else IMG_CH
    net = Policy(ch)
    n = sum(p.numel() for p in net.parameters())
    img = torch.randn(2, ch, IMG_SZ, IMG_SZ)
    q = torch.randn(2, QPOS)
    out = net(img, q)
    print(f"params: {n:,}  (target ~1M)")
    print("output:", tuple(out.shape), "range [", float(out.min()), ",", float(out.max()), "]")
    assert 0.5e6 < n < 2.0e6, n
    assert out.shape == (2, CHUNK, ACT)
    print("OK")
