"""THE CURRENT WAY -- the one source of truth for how data is made and read.

Every tool defaults to what is in this file, so the old way cannot come back
through a forgotten default.  This project has already been burned twice by a
default that pointed somewhere else: `arm_bench` accepted `--ds` and then
ignored it, replaying ds_0007 for a whole session while the belt ran at a
different speed; and `render_dataset` never set the line config, so it quietly
rendered the OLD stopping line and the jaws missed the cube.

WHAT THE OLD WAY WAS, AND WHY IT IS GONE
----------------------------------------
ds_0001..ds_0008 are not bad data, they are a DIFFERENT CONTRACT:

  * the arm started at HOME, not at the wait pose, so an episode began and
    ended in different places;
  * a "close" command stopped at GRIP_SHUT = 0.02 rad, which is a HALF-SHUT
    claw (0.02 is merely where the servo stalls on a 32 mm part) -- the new way
    commands the actuator's real lower bound, -0.17453 rad, and lets the
    +-2.94 N force limit stop it on the part;
  * the belt STOPPED at the indexing station; the new way the line NEVER stops
    (the user: "the assembly line shouldn't stop. it should always go");
  * only the manipulation window was rendered, so the belt wait -- the part of
    the episode the user actually specified -- was never rendered and the
    policy never saw a single wait frame.

Training on the old runs would teach the robot a world it will never meet, so
nothing in this file points at them.  The old datasets, caches and checkpoints
now live in _archive_old_way/; no tool reads them by default any more.

UPDATE 2026-10-03: the source of this file was accidentally deleted and is
recovered from its .pyc.  Defaults bumped to the 300-episode run: ds_0010 was
collected (303/306 ok), rendered (full episode, 303/303) on the running line;
cache_0010 and policy_0010_s12.pt are built from it below.
"""

# 2026-10-26: validated wrist-RGB/joint hybrid, unchanged c checkpoint.
# 200 fresh trials/condition: pick 199/198/199, strict place 198/195/199
# (nominal/tierB/boxX). See reports/visual_both_v12_final_validation_20261026.json.
DS = 'ds_0011'
CACHE = 'cache_0011'
CKPT = 'ckpt/policy_0041c.pt'
TRAIN_OUT = 'ckpt/policy_candidate.pt'  # never overwrite the validated default
RUNTIME_PROFILE = dict(dwell_mode='decision', place_settle_s=0.4,
                       release_primitive=True,visual_pick=True,visual_place=True,
                       visual_close_s=0.4,visual_takeover_z=0.81)
RUNNING_LINE = True
BELT_MPS = 0.08
BELT_FIXED = True
SPAWN_EVERY_S = 8.0
WAIT_JOINTS = (-1.44, -1.465, 0.3, 1.657, -1.449)
FULL_EPISODE = True
FPS = 25.0
