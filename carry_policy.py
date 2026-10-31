"""A phase-specific BC head; the existing policy still owns the visual encoder."""
import copy

import torch
from torch import nn


def carry_features(policy, image, qpos):
    visual = policy.pool(policy.b3(policy.b2(policy.b1(policy.stem(image))))).flatten(1)
    return torch.cat([visual, policy.prop(qpos)], dim=1)


class CarryHead(nn.Module):
    """Initialized identically to the base head, then trained on held-jaw rows.

    Enable only after proprioception confirms grasp and lift. Outputs retain
    the original 8x6 layout; runtime uses only arm channels from this head.
    """
    def __init__(self, policy):
        super().__init__()
        self.head = copy.deepcopy(policy.head)
        self.head.requires_grad_(True)

    def forward(self, features):
        return self.head(features).reshape(-1, 8, 6)
