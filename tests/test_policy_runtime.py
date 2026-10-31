"""Contract checks: camera parity and executed (not predicted) grip edges."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from policy_runtime import (GraspDwell, wrist_gray, temporal_actions,
                            PlacementMonitor, ReleaseOnDescent, carry_lookahead)


class RuntimeTests(unittest.TestCase):
    def test_carry_head_starts_identically_and_has_independent_weights(self):
        from model import Policy
        from carry_policy import CarryHead
        base = Policy(8, 12).eval()
        base.requires_grad_(False)
        head = CarryHead(base).eval()
        self.assertTrue(all(p.requires_grad for p in head.parameters()))
        features = torch.randn(2, 320)
        with torch.no_grad():
            self.assertTrue(torch.equal(head(features), base.head(features).view(2, 8, 6)))
            head.head[0].weight.zero_()
            self.assertFalse(torch.equal(head.head[0].weight, base.head[0].weight))

    def test_carry_lookahead_keeps_grip_timing_and_does_not_mutate_prediction(self):
        act = np.arange(48, dtype=np.float32).reshape(8, 6)
        saved = act.copy()
        out = carry_lookahead(act, 2, 3)
        np.testing.assert_array_equal(out[:, :5], act[3:5, :5])
        np.testing.assert_array_equal(out[:, 5], act[:2, 5])
        np.testing.assert_array_equal(act, saved)
        np.testing.assert_array_equal(carry_lookahead(act, 2, 7)[:, :5],
                                      act[[7, 7], :5])

    def test_optional_retention_protects_lift_off_and_resets_on_empty(self):
        ctl = ReleaseOnDescent(retain_on_stall=True)
        for now in (0, .04, .08, .12, .16, .20):
            ctl.step(now, [1], .2, True, [0, .245, .65])
        self.assertTrue(ctl.confirmed)
        self.assertFalse(ctl.lifted)
        _, shut = ctl.step(.24, [2], .2, False, [0, .245, .66])
        self.assertTrue(shut)
        self.assertIsNone(ctl.released_at)
        _, shut = ctl.step(.28, [2], -.1745, False, [0, .245, .66])
        self.assertFalse(shut)
        self.assertFalse(ctl.confirmed)

    def test_transient_closing_motion_does_not_arm_retention(self):
        ctl = ReleaseOnDescent(retain_on_stall=True)
        for i, gq in enumerate((.64, .4, .3, .2, .1, 0, -.1745)):
            ctl.step(i * .04, [1], gq, True, [0, .245, .65])
        self.assertFalse(ctl.confirmed)
        _, shut = ctl.step(.32, [2], -.1745, False, [0, -.15, .7])
        self.assertFalse(shut)
        self.assertIsNone(ctl.released_at)

    def test_release_requires_stall_lift_and_low_pose_away_from_lane(self):
        ctl = ReleaseOnDescent()
        for now in (0, .04, .08, .12, .16, .20):
            _, shut = ctl.step(now, [1], .2, True, [0, .245, .65])
            self.assertTrue(shut)
        self.assertTrue(ctl.confirmed)
        self.assertFalse(ctl.lifted)
        _, shut = ctl.step(.24, [2], .2, True, [0, .245, .82])
        self.assertTrue(ctl.lifted)
        _, shut = ctl.step(.28, [3], .2, False, [0, -.15, .82])
        self.assertTrue(shut)  # reject premature release during high transport
        _, shut = ctl.step(.32, [4], .2, True, [0, .245, .72])
        self.assertTrue(shut)  # low over belt is not a release
        arm, shut = ctl.step(.36, [5], .2, True, [0, -.15, .72])
        self.assertFalse(shut)
        arm, shut = ctl.step(.40, [6], .64, True, [0, -.15, .72])
        np.testing.assert_array_equal(arm, [5])
        self.assertFalse(shut)

    def test_empty_or_open_jaws_cannot_arm_release(self):
        for gq in (-.1745, .64):
            ctl = ReleaseOnDescent()
            for now in np.arange(0, 1, .04):
                ctl.step(now, [1], gq, True, [0, -.15, .85])
            self.assertFalse(ctl.confirmed)
            self.assertFalse(ctl.lifted)

    def test_placement_requires_release_and_uninterrupted_settling(self):
        monitor = PlacementMonitor(.4)
        self.assertFalse(monitor.step(0, True, True, 0))
        self.assertFalse(monitor.step(.1, True, False, .5))
        self.assertFalse(monitor.step(.2, True, False, .01))
        self.assertFalse(monitor.step(.4, False, False, .01))
        self.assertFalse(monitor.step(.6, True, False, .01))
        self.assertFalse(monitor.step(.8, True, False, .01))
        self.assertTrue(monitor.step(1., True, False, .01))
        self.assertEqual(monitor.entered_at, 0)

    def test_legacy_place_preserves_entry_only_metric(self):
        monitor = PlacementMonitor(0)
        self.assertFalse(monitor.step(0, False, False, 0))
        self.assertTrue(monitor.step(.04, True, True, 1))

    def test_temporal_decay_preserves_constant_commands(self):
        votes = {}
        act = np.full((8, 6), .6)
        for entry in range(0, 30, 2):
            out = temporal_actions(votes, act, entry, 2, .9)
            np.testing.assert_allclose(out, .6)
            self.assertTrue(all(k >= entry for k in votes))

    def test_temporal_alignment_and_newest_grip(self):
        votes = {}
        old = np.full((8, 6), .8)
        old[2:4, :5] = .4
        temporal_actions(votes, old, 0, 2, .5)
        new = np.full((8, 6), -.2)
        new[:, 5] = -1
        out = temporal_actions(votes, new, 2, 2, .5)
        np.testing.assert_allclose(out[:, :5], (.4 * .5 - .2) / 1.5,
                                   atol=1e-15)
        np.testing.assert_array_equal(out[:, 5], [-1, -1])

    def test_cache_decode_parity(self):
        rng = np.random.default_rng(20261026)
        frames = [np.full((128, 128, 3), v, np.uint8)
                  for v in (0, 1, 127, 254, 255)]
        frames.extend(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
                      for _ in range(20))
        for rgb in frames:
            t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
            expected = torch.round(F.interpolate(t, size=(96, 96),
                                                 mode="area")[0].mean(0)) / 255
            self.assertTrue(torch.equal(wrist_gray(rgb), expected))

    def test_close_in_second_chunk_entry_holds_preceding_command(self):
        dwell = GraspDwell(0.4)
        np.testing.assert_array_equal(dwell.step(0, [1, 2], False), [1, 2])
        np.testing.assert_array_equal(dwell.step(.04, [3, 4], True), [1, 2])
        np.testing.assert_array_equal(dwell.step(.40, [5, 6], True), [1, 2])
        np.testing.assert_array_equal(dwell.step(.48, [7, 8], True), [7, 8])
        self.assertAlmostEqual(dwell.until, .44)

    def test_open_cancels_hold_and_reclose_rearms(self):
        dwell = GraspDwell(.4)
        dwell.step(0, [1], False)
        dwell.step(.04, [2], True)
        np.testing.assert_array_equal(dwell.step(.08, [3], False), [3])
        np.testing.assert_array_equal(dwell.step(.12, [4], True), [3])
        self.assertAlmostEqual(dwell.until, .52)

    def test_zero_dwell_does_not_change_commands(self):
        dwell = GraspDwell(0)
        for now, shut in ((0, False), (.04, True), (.08, True), (.12, False)):
            np.testing.assert_array_equal(dwell.step(now, [now], shut), [now])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
