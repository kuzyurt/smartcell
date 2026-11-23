"""Label policy-visited carry states with an offline IK teacher.

The teacher's randomized tote goal is privileged TRAINING supervision only.
It never writes the robot controls or real MjData. Rollouts remain the current
wrist-image policy, and no boxX positions or evaluation seeds train.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import mujoco
import evaluate as EV
import collect_state as CS
import cam_variant as V
import build_cell as C
from collect_state import RandEnv
from conveyor_env import ConveyorPickEnv
from render_dataset import VisApply, sample_vis
from model import Policy, load_state
from carry_policy import carry_features
import gpu_guard
import ramcap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="ckpt/policy_0041c.pt")
    ap.add_argument("--n", type=int, default=16, help="episodes per training condition")
    ap.add_argument("--seed-base", type=int, default=1261000)
    ap.add_argument("--out", default="reports/carry_corrections_20261026.npz")
    args = ap.parse_args()
    if args.n <= 0 or Path(args.out).suffix != ".npz":
        ap.error("n must be positive and out must have .npz extension")
    if Path(args.out).exists():
        ap.error("output dataset already exists; choose a new --out to preserve training evidence")
    ramcap.install(threads=1)
    dev = gpu_guard.pick_cuda()
    ck = torch.load(args.ckpt, weights_only=True, map_location="cpu")
    model = Policy(int(ck["stack"]), int(ck["qdim"])).to(dev).eval()
    load_state(model, ck["model"], args.ckpt)
    EV.DWELL_MODE = "decision"
    EV.PLACE_SETTLE_S = .4
    EV.RELEASE_PRIMITIVE = True
    EV.CARRY_HEAD = None
    EV.CARRY_LOOKAHEAD = 0
    EV.RETAIN_ON_STALL = False
    EV.TE_M = 0
    # This historical corrective-data experiment labels the original base
    # policy's visited states, independent of the promoted deployment profile.
    EV.VISUAL_PICK = EV.VISUAL_PLACE = False
    rows = [json.loads(l) for l in open("ds_0011/manifest.jsonl") if l.strip()]
    CS.apply_line_config(rows[0]["line"])
    env = RandEnv(ConveyorPickEnv())
    r = env.real
    mount = V.Mount(r.model)
    visap = VisApply(r.model)
    rend = mujoco.Renderer(r.model, height=128, width=128)
    rend._scene_option.geomgroup[:] = [1] * 6
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    features, targets, splits, seeds, errors = [], [], [], [], []
    episodes = []
    for cond in ("nominal", "tierB"):
        rng = np.random.default_rng(args.seed_base + (cond == "tierB"))
        for i in range(args.n):
            start = len(features)
            last_sample = [-1.0]
            vis = sample_vis(rng) if cond == "tierB" else None
            st = {}

            def label(d):
                if (d["phase"] != "decision" or not d["carry_lifted"]
                        or d["primitive_release_t"] is not None
                        or d["t"] - last_sample[0] < .16):
                    return
                if r._grasped() is None:
                    return  # teacher labels genuinely held-object states only
                last_sample[0] = d["t"]
                site = r.data.site_xpos[r._grasp_site].copy()
                yaw = np.asarray(CS.CTX["jaw_yaw_vec"])
                center = np.asarray(C.TOTE_XY) + CS.jaw_center_xy(yaw)
                horizontal_error = float(np.linalg.norm(center - site[:2]))
                if horizontal_error > .025:
                    goal = np.r_[site[:2] if site[2] < .79 else center, .82]
                else:
                    goal = np.r_[center, C.TOTE_RELEASE_Z]
                delta = goal - site
                target = site + delta * min(1., .18 * .32 / max(1e-8, np.linalg.norm(delta)))
                q0 = r.data.qpos[r._arm_qpos].copy()
                # IK owns a PRIVATE MjData. Assert real state stays untouched.
                real_state = r.data.qpos.copy()
                r.ik.d.qpos[r.ik.grip] = r.data.qpos[r._grip_qpos]
                qgoal, err, _ = r.ik.solve(target, q0=q0, yaw=yaw,
                                          outer=6, iters=10, restarts=0)
                if not np.array_equal(real_state, r.data.qpos):
                    raise RuntimeError("teacher mutated real robot state")
                if not np.isfinite(err) or err > .008:
                    return
                qgoal = q0 + np.clip(qgoal - q0, -.6, .6)
                raw = q0 + np.arange(1, 9)[:, None] / 8 * (qgoal - q0)
                action = np.full((8, 6), -1., np.float32)
                action[:, :5] = np.clip((raw - r._mid) / r._half, -1, 1)
                prop = torch.from_numpy(d["q6"].astype(np.float32))[None].to(dev)
                with torch.inference_mode():
                    f = carry_features(model, d["policy_image"], prop)[0].cpu().numpy()
                features.append(f.copy())
                targets.append(action)
                splits.append(int(i % 5 == 4))  # complete episodes held out
                errors.append(float(err))

            EV.rollout(env, model, mount, rend, visap, vis, rng, int(ck["stack"]),
                       lrng=np.random.default_rng(args.seed_base + 777 + i),
                       on_step=label, stats=st)
            seeds.extend([st["seed"]] * (len(features) - start))
            episodes.append(dict(cond=cond, i=i, split=int(i % 5 == 4),
                                 samples=len(features) - start, **st))
            print(cond, i, "samples", len(features) - start, "pick", st["picked"],
                  "place", st["placed"], flush=True)
    if not features:
        raise RuntimeError("no admissible carry supervision collected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(args=vars(args), base_sha256=hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
                    kind="policy_visited_ik_training_labels", privileged_labels=True,
                    oracle_controls_robot=False, boxX_used=False,
                    episodes=episodes,
                    hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in (__file__, os.path.join(os.path.dirname(os.path.abspath(__file__)), "carry_policy.py"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))})
    np.savez_compressed(out, features=np.stack(features), targets=np.stack(targets),
                        split=np.asarray(splits), seeds=np.asarray(seeds),
                        ik_error=np.asarray(errors), metadata=json.dumps(metadata))
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print("saved", len(features), "teacher-labelled states to", out, flush=True)
    rend.close()
    r.close()


if __name__ == "__main__":
    main()
