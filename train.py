"""Phase 4 trainer: behaviour cloning with action chunking.

Loss: Huber (delta=1) on all 8 x 6 chunk entries.  Optimizer AdamW, cosine
schedule, AMP on the 2060.  Val = same loss on held-out episodes.  Checkpoint
is selected on val loss, and the test split is never touched here.

Run:  python train.py [--epochs 40] [--batch 256]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

print("mujoco (train.py) torch", torch.__version__)

import ramcap
from model import Policy
import dataset as D
from dataset import make_loaders
import current


GRIP_W = 1.0
"""Weight on the gripper channel (index 5) in the TRAIN loss.

E1: grip is 9.2% of loss, transitions 3% of rows / 7% of gradient - the
close trigger is a whisper. GRIP_W multiplies the grip entries of the
elementwise Huber before averaging. Val/test ALWAYS use the unweighted
loss so runs stay comparable. 1.0 = off (default recipe)."""


def loss_fn(pred, target):
    l = nn.functional.huber_loss(pred, target, reduction="none", delta=1.0)
    if GRIP_W != 1.0:
        l = l.clone()
        l[..., 5] *= float(GRIP_W)
    return l.mean()


def evaluate(model, loader, dev):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for img, q, act in loader:
            img, q, act = img.to(dev), q.to(dev), act.to(dev)
            # unweighted by construction: val compares runs, it does not train
            tot += float(nn.functional.huber_loss(model(img, q), act,
                                                  delta=1.0)) * len(img)
            n += len(img)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stack", type=int, default=8,
                    help="history length in frames (at 25 Hz: 8 = 280 ms)")
    ap.add_argument("--aug-sigma", type=float, default=0.015,
                    help="IST/SOLL noise (rad).  The demos hold ONE arm pose "
                         "for the whole 8 s wait, so without this the policy "
                         "is trained on a zero-width manifold and diverges "
                         "within 0.4 s of its own servo noise.")
    ap.add_argument("--aug-rot-deg", type=float, default=2.0,
                    help="wrist-view rotation augmentation (deg); what a "
                         "wrist offset looks like to a wrist-mounted camera")
    ap.add_argument("--aug-gap-k", type=float, default=3.0,
                    help="DAgger-lite: scale the SOLL-IST gap by U(0, K). "
                         "The rollout feeds the policy's own command back as "
                         "SOLL, creating gap bands the expert never visits "
                         "(0.3-1.2 rad) - exactly where the lunge dies.")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers.  Each one is a full spawned "
                         "interpreter inside the RAM cap, so this is a memory "
                         "knob as much as a speed knob; 0 = in-process.")
    ap.add_argument("--extra-ds", default=None,
                    help="DAgger corrections ds dir (dagger_collect.py); "
                         "joined to TRAIN ONLY, val/test stay --ds.  Comma-"
                         "separated to join several rounds at once "
                         "(ds_0010_d1,ds_0010_d2); order matches --extra-cache")
    ap.add_argument("--extra-cache", default=None,
                    help="cache dirs built from --extra-ds by build_cache.py, "
                         "comma-separated, same order")
    ap.add_argument("--out", default=current.TRAIN_OUT)
    ap.add_argument("--ds", default=current.DS,
                    help="episode manifest dir (required pairing with --cache)")
    ap.add_argument("--cache", default=current.CACHE,
                    help="cache dir built from --ds by build_cache.py")
    ap.add_argument("--init", default=None,
                    help="warm-start weights from this ckpt (same stack/qdim) "
                         "with a fresh optimizer/schedule for --epochs more. "
                         "Use to continue a run that hit the wall-clock limit.")
    ap.add_argument("--start-epoch", type=int, default=0,
                    help="global epoch offset for --keep-all filenames when "
                         "continuing with --init (prevents overwriting the "
                         "first run's .eN.pt files and keeps best-val "
                         "comparison honest across the continuation).")
    ap.add_argument("--drop-p", type=float, default=0.0,
                    help="proprio dropout prob (train only, E2 remedy). "
                         "0.5 = half the rows vision-only.")
    ap.add_argument("--trans-p", type=float, default=0.0,
                     help="fraction of train samples drawn from "
                          "grip-transition rows (E1 remedy). 0.25 suggested.")
    ap.add_argument("--desc-p", type=float, default=0.0,
                     help="fraction of train samples drawn from pre-release "
                          "descent rows (hover-stall remedy: the last 32 rows "
                          "before the release edge). 0.30 suggested.")
    ap.add_argument("--grip-w", type=float, default=1.0,
                    help="train-loss weight on the grip channel (E1 remedy). "
                         "Val stays unweighted. 3.0 suggested.")
    ap.add_argument("--keep-all", action="store_true",
                    help="save every epoch to <out>.e<epoch>.pt so the best "
                         "checkpoint can be selected by PICK/PLACE rate "
                         "(evaluate.py --n 24 quick gate), not by val loss. "
                         "Val loss is a proxy; pick+place is the metric. "
                         "See select_best.py.")
    args = ap.parse_args()

    ramcap.install()
    D.set_stack(args.stack)
    D.AUG_SIGMA = args.aug_sigma
    D.AUG_ROT_DEG = args.aug_rot_deg
    D.AUG_GAP_K = args.aug_gap_k
    D.AUG_DROP_P = args.drop_p
    D.P_TRANS = args.trans_p
    D.P_DESC = args.desc_p
    global GRIP_W
    GRIP_W = float(args.grip_w)
    if args.drop_p or args.trans_p or args.desc_p or float(args.grip_w) != 1.0:
        print(f"E1/E2 remedies: proprio-drop p={args.drop_p:g}, "
              f"trans-sample p={args.trans_p:g}, desc-sample p={args.desc_p:g}, "
              f"grip-w {float(args.grip_w):g} "
              f"(val stays unweighted)", flush=True)
    print(f"history: {args.stack} frames = "
          f"{(args.stack - 1) * D.GRID_MS} ms lookback at "
          f"{1000 // D.GRID_MS} Hz")

    import gpu_guard
    import os as _os2
    if _os2.environ.get("SMARTCELL_ALLOW_CPU", "0") == "1" and not torch.cuda.is_available():
        dev = "cpu"
        print("device: cpu (SMARTCELL_ALLOW_CPU=1 override)", flush=True)
    else:
        dev = gpu_guard.pick_cuda()

    if args.extra_ds and args.extra_cache:
        eds_l = [s for s in args.extra_ds.split(",") if s]
        eca_l = [s for s in args.extra_cache.split(",") if s]
        assert len(eds_l) == len(eca_l), \
            "--extra-ds and --extra-cache must have the same length"
        extra = list(zip(eds_l, eca_l))
    else:
        assert not (args.extra_ds or args.extra_cache), \
            "--extra-ds and --extra-cache go together"
        extra = None
    tr, va, _ = make_loaders(args.ds, args.cache, batch=args.batch,
                             stack=args.stack, workers=args.workers,
                             extra=extra)
    # The proprio width comes from the DATA, not a constant.  Reading it off the
    # cache means a 6-wide (pre-SOLL) cache trains a 6-wide policy instead of
    # dying on a shape mismatch deep in the head, and it is impossible to train
    # a SOLL model on a cache that has no SOLL in it.
    ep0 = tr.dataset.eps[0]
    ist_w, soll_w = int(ep0.q.shape[1]), int(ep0.soll.shape[1])
    qdim = ist_w + soll_w
    print(f"proprio: {qdim} wide = IST {ist_w} + SOLL {soll_w}")
    model = Policy(img_ch=args.stack, qdim=qdim).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    if args.init:
        from model import load_state as _ls
        _ck = torch.load(args.init, weights_only=True, map_location=dev)
        assert int(_ck.get("stack", args.stack)) == args.stack, \
            f"--init stack {_ck.get('stack')} != --stack {args.stack}"
        assert int(_ck.get("qdim", qdim)) == qdim, \
            f"--init qdim {_ck.get('qdim')} != data qdim {qdim}"
        _ls(model, _ck["model"], args.init)
        print(f"init: loaded weights from {args.init} "
              f"(epoch {_ck.get('epoch')}, val {_ck.get('val')}) "
              f"fresh optimizer/schedule for {args.epochs} more epochs")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    # fp16 + GradScaler: measured 53.5 ms/batch vs 367 ms for bf16 on this
    # Turing card (no bf16 tensor cores - software emulation is 7x slower)
    import gpu_guard as _gg
    AMP = _gg.cuda_amp_dtype()
    use_amp = (dev != "cpu")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best = float("inf")
    if args.init and args.start_epoch <= 0:
        print("WARNING: --init without --start-epoch: --keep-all files "
              "(.eN.pt) will OVERWRITE the first run's epochs and best-val "
              "restarts at inf (a worse model can clobber --out). Pass "
              "--start-epoch N to continue numbering.", flush=True)
    os_dir = "/".join(args.out.split("/")[:-1])
    import os
    if os_dir:            # a bare --out name means cwd; makedirs("") raises
        os.makedirs(os_dir, exist_ok=True)
    for ep in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        t0 = time.perf_counter()
        tot, n = 0.0, 0
        for img, q, act in tr:
            img, q, act = img.to(dev, non_blocking=True), q.to(dev), act.to(dev)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=AMP):
                    loss = loss_fn(model(img, q), act)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(img, q), act)
                loss.backward()
                opt.step()
            tot += float(loss) * len(img)
            n += len(img)
        sched.step()
        ramcap.guard(f"train epoch {ep}")
        vl = evaluate(model, va, dev)
        star = ""
        if vl < best:
            best = vl
            star = " *"
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val": vl, "train": tot / n,
                        "stack": args.stack, "img_ch": args.stack,
                        "qdim": qdim,
                        "aug_sigma": args.aug_sigma,
                        "aug_rot_deg": args.aug_rot_deg,
                        "aug_gap_k": args.aug_gap_k,
                        "drop_p": args.drop_p,
                         "trans_p": args.trans_p,
                         "desc_p": args.desc_p,
                         "grip_w": float(args.grip_w),
                         # WHICH dataset.  A checkpoint loaded against the
                        # wrong dataset looks fine and behaves badly: the
                        # inspector on :8766 was showing policy_0005_s12
                        # (trained on ds_0005: the fabricated camera mount and
                        # the line that STOPPED at the station) next to ds_0007
                        # frames, and the user read the policy's failure to
                        # grasp as a broken example.
                        "ds": args.ds,
                        "extra": ([f"{d}+{c}" for d, c in extra]
                                  if extra else None)}, args.out)
        print(f"  epoch {ep:3d}  train {tot / n:.5f}  val {vl:.5f}{star}  "
              f"({time.perf_counter() - t0:.1f}s)"
              f"  ram {ramcap.guard(f'epoch {ep}'):.2f} proc / "
              f"{ramcap.job_usage()[1]:.2f} tree peak"
              f" / {ramcap.CAP_GB:.0f} GB cap",
              flush=True)

        if args.keep_all:
            ep_path = f"{args.out}.e{ep}.pt"
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val": vl, "train": tot / n,
                        "stack": args.stack, "img_ch": args.stack,
                        "qdim": qdim,
                        "aug_sigma": args.aug_sigma,
                        "aug_rot_deg": args.aug_rot_deg,
                        "aug_gap_k": args.aug_gap_k,
                        "drop_p": args.drop_p,
                         "trans_p": args.trans_p,
                         "desc_p": args.desc_p,
                         "grip_w": float(args.grip_w),
                         "ds": args.ds,
                        "extra": ([f"{d}+{c}" for d, c in extra]
                                  if extra else None)}, ep_path)

    print(f"\nbest val {best:.5f} -> {args.out}")
    print("checkpoint layout: model.state_dict, epoch, val, train, stack, "
          "img_ch, qdim")
    print("NOTE: pick+place rate outranks val loss. Run evaluate.py --n 24 "
          "quick gate on --keep-all epochs (or top-3 val epochs) and pick "
          "with select_best.py (place, then pick, then val).")


if __name__ == "__main__":
    main()
