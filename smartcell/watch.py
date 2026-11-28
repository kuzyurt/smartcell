"""Live viewer: watch the trained BC policy drive, and see what it actually sees.

Runs the EXACT evaluation rollout (evaluate.rollout, unmodified - it is handed
an on_step observer) and serves a montage to a browser at http://127.0.0.1:8765.

The montage is four things side by side, so the gap between "the cell" and
"the network's input" is impossible to miss:

  1. overhead + preview   the cell from two human cameras (world truth)
  2. wrist RGB 128x128    what the wrist camera renders  (mujoco.Renderer)
  3. MODEL INPUT 96x96    grey = channel mean, /255 - this is literally the
                          tensor evaluate.rollout feeds the net (x2 nearest)
  4. the sampled stack    every frame the network actually reads at this
                          decision, including its independent frame latency

Run:  python smartcell/watch.py
      python smartcell/watch.py --ckpt ckpt/policy_0041c.pt --condition tierB
      python smartcell/watch.py --episodes 3 --save frames/
"""
import argparse
import base64
import io
import json
import os
import threading
import time

import numpy as np
import mujoco
import torch

from PIL import Image, ImageDraw, ImageFont

import build_cell as C
import cam_variant as V
import evaluate as EV
import collect_state as CS
from render_dataset import sample_vis, VisApply
from collect_state import RandEnv
from conveyor_env import ConveyorPickEnv
from model import Policy, load_state
import current

BG = (14, 16, 20)
FG = (232, 236, 240)
DIM = (128, 138, 150)
ACC = (255, 186, 60)
GOOD = (90, 220, 140)
BAD = (255, 96, 96)


def _font(sz):
    for p in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


F_TITLE = _font(17)
F_LAB = _font(13)
F_SM = _font(11)


def _fit_font(sz):
    return _font(sz)


# --------------------------------------------------------------------------
# montage layout (pixels).  Panel boxes are (x, y, w, h) of the IMAGE itself;
# labels sit 26 px above each.
PANELS = [
    ("overhead camera  (world)", "overhead", 320, 240),
    ("preview camera  (world)", "preview", 320, 240),
    ("wrist_cam RGB  128x128", "wrist", 256, 256),
    ("MODEL INPUT  96x96 grey", "model", 192, 192),
    ("FRAME STACK  (96x96 each, newest left)", "stack", 586, 194),
]
PAD, GAP, TOP = 10, 10, 64
MONT_W = PAD * 2 + sum(p[2] for p in PANELS) + GAP * (len(PANELS) - 1)
MONT_H = TOP + 290


def compose(world_over, world_pre, wrist_rgb, model_gray, stack, status):
    """stack: list of (H,W) uint8 in [-1 - k] order, newest first."""
    im = Image.new("RGB", (MONT_W, MONT_H), BG)
    d = ImageDraw.Draw(im)
    d.text((PAD, 8), status, font=F_TITLE, fill=FG)

    x = PAD
    for (label, _kind, w, h) in PANELS:
        d.text((x, TOP - 18), label, font=F_LAB, fill=ACC)
        x += w + GAP

    x = PAD
    for (label, kind, w, h) in PANELS:
        y = TOP
        if kind == "stack":
            # 6 x 2 grid of true-resolution frames, 2 px gutters, newest first
            cols, rows = 6, 2
            fh = fw = 96
            for i, fr in enumerate(stack[:cols * rows]):
                cx = x + (i % cols) * (fw + 2)
                cy = y + (i // cols) * (fh + 2)
                t = Image.fromarray(fr).resize((fw, fh), Image.NEAREST)
                im.paste(t, (cx, cy))
                # outline ONLY, drawn outside the tile: anything painted on
                # top of the pixels would stop this panel being a faithful
                # view of the net's input
                d.rectangle([cx - 1, cy - 1, cx + fw, cy + fh],
                            outline=(ACC if i == 0 else (52, 58, 66)))
            d.text((x, y + 2 * fh + 6),
                   f"{len(stack)} sampled frames, newest left; {(len(stack)-1)*40} ms nominal lookback"
                   "   |   independent 0-40 ms frame staleness",
                   font=F_SM, fill=DIM)
        else:
            src = {"overhead": world_over, "preview": world_pre,
                   "wrist": wrist_rgb, "model": model_gray}[kind]
            t = Image.fromarray(src)
            # NEAREST everywhere: this is a pixel-truth view, do not invent detail
            im.paste(t.resize((w, h), Image.NEAREST), (x, y))
            d.rectangle([x - 1, y - 1, x + w, y + h], outline=(52, 58, 66))
            if kind == "model":
                d.text((x, y + h + 6),
                       "grey /255; last policy input",
                       font=F_SM, fill=DIM)
            if kind == "wrist":
                d.text((x, y + h + 6),
                       "live RGB; control uses sampled history",
                       font=F_SM, fill=DIM)
        x += w + GAP
    return im


# --------------------------------------------------------------------------
def wrist_rgb(env, mount, rend_rgb):
    """The 128x128 RGB image BEFORE the grey/resize that becomes the model input.

    Byte-for-byte the same source evaluate.current_frame averages - verified in
    self_test(): decoding this panel reproduces the net's input tensor exactly.
    """
    mount.update(env.real.data)
    rend_rgb.update_scene(env.real.data,
                          camera=int(env.real.model.camera("wrist_cam").id))
    V.strip_scene_markers(rend_rgb)
    return rend_rgb.render()


class _Restart(Exception):
    """Raised inside on_step when the web RESTART button aborts an episode."""


class Viewer:
    def __init__(self, args):
        EV.ramcap.install(threads=1)
        self.args = args
        self.lock = threading.Lock()
        self.png = None          # bytes of the latest montage
        self.info = {}
        self.log = []
        self.ep_i = 0
        self.done = threading.Event()
        self.restart = threading.Event()   # set by the web RESTART button

        import gpu_guard
        dev = gpu_guard.pick_cuda()
        ck = torch.load(args.ckpt, weights_only=True, map_location=dev)
        self.stack = int(ck.get("stack", EV.STACK))
        # qdim from the ckpt: 6 for pre-SOLL checkpoints, 12 for IST||SOLL ones
        self.qdim = int(ck.get("qdim", 6))
        self.model = Policy(img_ch=self.stack, qdim=self.qdim).to(dev)
        load_state(self.model, ck["model"], args.ckpt)
        self.model.eval()
        self.ck_info = (f"{args.ckpt}  epoch {ck['epoch']}  val {ck['val']:.5f}"
                        f"  stack {self.stack}  proprio {self.qdim}"
                        f" ({'IST+SOLL' if self.qdim > 6 else 'IST only'})")
        if EV.VISUAL_PICK or EV.VISUAL_PLACE:
            phases='/'.join(name for name,active in
                            (('pick',EV.VISUAL_PICK),('place',EV.VISUAL_PLACE)) if active)
            self.ck_info += f'  wrist RGB alignment: {phases}'

        # Match the evaluator's recorded running-line physics. Without this
        # the live viewer silently used the old stopping line.
        with open(os.path.join(args.ds, "manifest.jsonl")) as source:
            first = next(json.loads(row) for row in source if row.strip())
        CS.apply_line_config(first.get("line"))
        self.env = RandEnv(ConveyorPickEnv())
        self.r = self.env.real
        self.m = self.r.model
        self.visap = VisApply(self.m)
        self.mount = V.Mount(self.m)

        def mk(h, w):
            rd = mujoco.Renderer(self.m, height=h, width=w)
            rd._scene_option.geomgroup[:] = [1] * 6
            rd.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            return rd

        # SAME height/width/flags as evaluate.py's renderer -> the wrist panel
        # is the exact pre-processing source, not a lookalike
        self.rend_ev = mk(128, 128)
        self.rend_over = mk(PANELS[0][2], PANELS[0][3])
        self.rend_pre = mk(PANELS[1][2], PANELS[1][3])
        self.wrist_id = int(self.m.camera("wrist_cam").id)
        self.over_id = int(self.m.camera("overhead").id)
        self.pre_id = int(self.m.camera("preview").id)

    # -- render helpers ---------------------------------------------------
    def world(self):
        self.rend_over.update_scene(self.r.data, camera=self.over_id)
        self.rend_pre.update_scene(self.r.data, camera=self.pre_id)
        return self.rend_over.render(), self.rend_pre.render()

    def close(self):
        for rd in (self.rend_ev, self.rend_over, self.rend_pre):
            try:
                rd.close()
            except Exception:
                pass
        self.r.close()

    # -- the episode loop -------------------------------------------------
    def _blank(self):
        return dict(t=0.0, gap=0.0, grip=0.0, arm=[0.0] * 5, site=[0.0] * 3,
                    part=[0.0] * 3, q6=[0.0] * 6, grasped=False, picked=False,
                    placed=False, raw=[], cmd=[], phase="start", n=0)

    def run(self):
        a = self.args
        rng = np.random.default_rng(a.seed)
        boxes = EV.BOX_X_CORNERS
        n_ep = a.episodes
        i = 0
        picks = places = 0
        while not self.done.is_set() and (n_ep <= 0 or i < n_ep):
            cond = a.condition
            vis = sample_vis(rng) if cond == "tierB" else None
            box_off = (tuple(boxes[i % len(boxes)])
                       if cond == "boxX" else None)
            try:
                picked, placed = self._one(i, cond, vis, rng, box_off)
            except _Restart:
                # button pressed mid-episode: drop it, start a fresh one
                with self.lock:
                    self.info["status"] = f"episode {i + 1} restarted by user"
                self.restart.clear()
                continue
            i += 1
            self.ep_i = i
            picks += int(picked)
            places += int(placed)
            print(f"  {cond} ep {i}" + (f"/{n_ep}" if n_ep > 0 else "") +
                  f"  pick {int(picked)}  place {int(placed)}"
                  f"   running {picks}/{i} pick, {places}/{i} place", flush=True)
            self.log.append(dict(ep=i, cond=cond, pick=bool(picked),
                                 place=bool(placed)))
            time.sleep(a.pause)
        with self.lock:
            self.info = dict(status="finished", ck=self.ck_info,
                             log=self.log[-12:])
        self.done.set()

    def _one(self, i, cond, vis, rng, box_off):
        self.cur = self._blank()
        self.policy_stack = None
        self.cur_cond = cond
        self.ep_i = i
        st = dict(n=0, t0=time.perf_counter(), t_last=time.perf_counter())

        def on_step(s):
            if self.done.is_set():
                return
            if self.restart.is_set():      # web button: abort this episode now
                raise _Restart()
            st["n"] += 1
            ph = s["phase"]
            if ph == "wait":
                # Belt approach.  Nobody needs to watch the part travel 1.4 m,
                # so paint rarely and NEVER sleep here (a sleep would make the
                # belt crawl).  1 paint per 8 frames = 1 per 320 ms of sim.
                self.cur["phase"] = "wait"
                self.cur["n"] = st["n"]
                if st["n"] % 8 == 0:
                    self._paint(s, self.cur_cond, i, st["t_last"] - st["t0"])
                return
            if ph == "policy":
                self.cur.update(t=s["t"], gap=s["gap_mm"], grip=s["grip"],
                                arm=[float(v) for v in s["arm"]],
                                site=[float(v) for v in s["site"]],
                                part=[float(v) for v in s["part"]],
                                q6=[float(v) for v in s["q6"]],
                                grasped=bool(s["grasped"]),
                                picked=bool(s["picked"]),
                                placed=bool(s["placed"]))
            elif ph == "decision":
                self.policy_stack = np.clip(s['policy_image'][0].detach().cpu().numpy()*255,
                                            0,255).astype(np.uint8)
                self.cur["raw"] = [float(v) for v in s["raw_grip"]]
                self.cur["cmd"] = [float(v) for v in s["commanded"]]
            self.cur["phase"] = ph
            self.cur["n"] = st["n"]
            # pace the wall clock so the rollout is watchable at --fps
            dt = time.perf_counter() - st["t_last"]
            if dt < self.args.period:
                time.sleep(self.args.period - dt)
            st["t_last"] = time.perf_counter()
            self._paint(s, self.cur_cond, i, st["t_last"] - st["t0"])

        return EV.rollout(self.env, self.model, self.mount, self.rend_ev,
                          self.visap, vis, rng, self.stack, None,
                          lrng=np.random.default_rng(777 + i), box=box_off,
                          on_step=on_step)

    def _paint(self, s, cond, ep, wall):
        """Compose + publish one montage.  Called from inside the rollout."""
        hist = s["hist"]                      # list of torch (96,96) float32
        n = len(hist)

        def u8(i):
            return np.clip(hist[i].numpy() * 255.0, 0, 255).astype(np.uint8)

        cur = u8(n - 1)
        # during the belt wait the buffer is still filling and can be shorter
        # than `stack`; pad at the OLD end with the earliest real frame, the
        # same thing evaluate.rollout does before its first decision
        stack = [u8(n - 1 - k if 1 + k <= n else 0)
                 for k in range(self.stack)]
        if self.policy_stack is not None:
            stack = list(self.policy_stack)
            cur = stack[0]
        over, pre = self.world()
        wrgb = wrist_rgb(self.env, self.mount, self.rend_ev)
        c = self.cur
        pill = ("PICKED " if c.get("picked") else "") + \
               ("PLACED" if c.get("placed") else "")
        if c.get("phase") == "wait":
            status = (f"episode {ep}   {cond}   --  belt running, the arm "
                      f"holds at hover and waits for the approaching part --")
        else:
            status = (f"t = {c.get('t', 0.0):6.2f} s   ep {ep}   {cond}   "
                      f"jaw {c.get('gap', 0.0):5.1f} mm   "
                      f"grip {'SHUT' if c.get('grip', 0) < 0 else 'OPEN'}   {pill}")
        img = compose(over, pre, wrgb, cur, stack, status)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=self.args.quality)
        with self.lock:
            self.png = base64.b64encode(buf.getvalue()).decode("ascii")
            self.info = dict(
                status=status, ck=self.ck_info, cond=cond, ep=ep,
                t=round(float(c.get("t", 0.0)), 3),
                gap=round(float(c.get("gap", 0.0)), 1),
                grip=round(float(c.get("grip", 0.0)), 4),
                arm=[round(float(v), 4) for v in c.get("arm", [0] * 5)],
                site=[round(float(v) * 1000, 1) for v in c.get("site", [0] * 3)],
                part=[round(float(v) * 1000, 1) for v in c.get("part", [0] * 3)],
                q6=[round(float(v), 4) for v in c.get("q6", [0] * 6)],
                raw=[round(float(v), 3) for v in c.get("raw", [])],
                cmd=[round(float(v), 3) for v in c.get("cmd", [])],
                picked=bool(c.get("picked")), placed=bool(c.get("placed")),
                grasped=bool(c.get("grasped")),
                n=c.get("n", 0), wall=round(wall, 2) if wall else None,
                log=self.log[-12:])
        if self.args.save:
            os.makedirs(self.args.save, exist_ok=True)
            img.save(os.path.join(self.args.save,
                                  f"ep{ep:03d}_{c.get('n', 0):05d}.png"))


# --------------------------------------------------------------------------
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>smartcell - policy live view</title>
<style>
 :root{--bg:#0e1014;--fg:#e8ecf0;--dim:#889;--acc:#ffba3c;--good:#5adc8c;--bad:#ff6060}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 Consolas,Menlo,monospace}
 h1{font-size:15px;margin:0;padding:10px 14px 4px;color:var(--acc);letter-spacing:.4px}
 .sub{padding:0 14px 10px;color:var(--dim);font-size:12px}
 #wrap{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:12px;padding:0 14px 18px}
 #pic{width:100%;height:auto;display:block;border:1px solid #232830;border-radius:6px;background:#000}
 #side{background:#151920;border:1px solid #232830;border-radius:6px;padding:12px;height:max-content}
 .row{display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-bottom:1px solid #1e232b}
 .k{color:var(--dim)} .v{text-align:right}
 h2{font-size:12px;color:var(--acc);margin:14px 0 6px;letter-spacing:.6px;text-transform:uppercase}
 .bar{height:9px;background:#222833;border-radius:2px;position:relative;margin:3px 0 7px}
 .bar i{position:absolute;top:0;height:100%;background:var(--acc);border-radius:2px}
 .bar u{position:absolute;top:-2px;width:2px;height:13px;background:var(--fg)}
 .lbl{display:flex;justify-content:space-between;color:var(--dim);font-size:11px}
 .pill{padding:1px 7px;border-radius:9px;font-size:11px;font-weight:bold}
 .on{background:#1d3a2a;color:var(--good)} .off{background:#2a1d1d;color:var(--bad)}
 #log{font-size:11px;color:var(--dim);white-space:pre;margin-top:6px;max-height:150px;overflow:auto}
 #rst{margin-left:10px;background:#232833;color:var(--fg);border:1px solid #3a4250;border-radius:5px;padding:3px 12px;font:12px Consolas,monospace;cursor:pointer}
 #rst:hover{background:#2f3642;border-color:var(--acc)}
</style></head><body>
<h1 id="h">smartcell &mdash; live policy</h1>
<div class="sub" id="sub">connecting... <button id="rst" onclick="fetch('/restart')" title="abort this episode and start a fresh one">&#8635; restart episode</button></div>
<div id="wrap">
 <img id="pic" alt="waiting">
 <div id="side">
  <h2>robot state</h2>
  <div class="row"><span class="k">sim time</span><span class="v" id="t">-</span></div>
  <div class="row"><span class="k">jaw gap</span><span class="v" id="gap">-</span></div>
  <div class="row"><span class="k">grip</span><span class="v" id="grip">-</span></div>
  <div class="row"><span class="k">grasping</span><span class="v" id="gr">-</span></div>
  <div class="row"><span class="k">picked</span><span class="v" id="pk">-</span></div>
  <div class="row"><span class="k">placed</span><span class="v" id="pl">-</span></div>
  <h2>proprio qpos (the net's other input)</h2>
  <div id="q6" class="lbl"><span>-</span></div>
  <h2>arm target vs actual (norm -1..1)</h2>
  <div id="arms"></div>
  <h2>predicted grip over the 8-entry chunk</h2>
  <div id="raw" class="lbl"><span>-</span></div>
  <h2>decoded policy grip (alignment may override)</h2>
  <div id="cmd" class="lbl"><span>-</span></div>
  <h2>last episodes</h2><div id="log"></div>
 </div>
</div>
<script>
const $=i=>document.getElementById(i);
const fmt=v=>v===undefined?'-':(typeof v==='number'?(Math.abs(v)<1e4?v.toFixed(3):v):v);
function bars(el,vals,cols){
  const max=Math.max(1,...vals.map(Math.abs));
  $(el).innerHTML=vals.map((v,i)=>{
    const w=Math.abs(v)/max*50;
    const left=v>=0?50:50-w;
    return `<div class="lbl"><span>${cols[i]}</span><span>${v.toFixed(2)}</span></div>
            <div class="bar"><i style="left:${left}%;width:${w}%;background:${Math.abs(v)>0.85?'#5adc8c':'#ffba3c'}"></i><u style="left:50%"></u></div>`;
  }).join('');
}
function ticks(el,vals){
  $(el).innerHTML=vals.map(v=>{
    const p=Math.max(0,Math.min(100,(v+1)/2*100));
    return `<div class="lbl"><span>|</span><span>${v.toFixed(2)}</span></div><div class="bar"><i style="left:${p}%;width:2px"></i></div>`;
  }).join('');
}
function poll(){
 fetch('/state',{cache:'no-store'}).then(r=>r.json()).then(d=>{
  if(!d.status){$('sub').textContent='waiting for the first frame...';return setTimeout(poll,300);}
  $('sub').textContent=d.ck+'   |   '+(d.wall!==null?d.wall.toFixed(2)+' s of wall clock so far':'');
  $('h').textContent='smartcell — live policy';
  if(d.img) $('pic').src='data:image/jpeg;base64,'+d.img;
  $('t').textContent=d.t.toFixed(2)+' s';
  $('gap').textContent=d.gap.toFixed(1)+' mm';
  $('grip').innerHTML='<span class="pill '+(d.grip<0?'off':'on')+'">'+(d.grip<0?'SHUT':'OPEN')+'</span>';
  $('gr').textContent=d.grasped?'yes':'no';
  $('pk').innerHTML='<span class="pill '+(d.picked?'on':'off')+'">'+(d.picked?'YES':'no')+'</span>';
  $('pl').innerHTML='<span class="pill '+(d.placed?'on':'off')+'">'+(d.placed?'YES':'no')+'</span>';
  $('q6').innerHTML=d.q6.map((v,i)=>`<span>${fmt(v)}</span>`).join('');
  bars('arms',d.arm,['x','y','z','rx','ry']);
  if(d.raw&&d.raw.length) ticks('raw',d.raw); else $('raw').innerHTML='<span>-</span>';
  if(d.cmd&&d.cmd.length) ticks('cmd',d.cmd); else $('cmd').innerHTML='<span>-</span>';
  $('log').textContent=(d.log||[]).map(l=>`ep${l.ep}  ${l.cond}  pick ${l.pick?1:0}  place ${l.place?1:0}`).join('\n');
  setTimeout(poll, 60);
 }).catch(()=>setTimeout(poll,700));
}
poll();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=current.CKPT)
    ap.add_argument("--ds", default=current.DS,
                    help="manifest supplying the same line configuration as evaluation")
    ap.add_argument("--condition", default="nominal",
                    choices=["nominal", "tierB", "boxX"])
    ap.add_argument("--episodes", type=int, default=0, help="0 = endless")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--pause", type=float, default=1.5,
                    help="seconds between episodes")
    ap.add_argument("--save", default=None, help="dump montages to this dir")
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    a.period = 1.0 / max(0.5, a.fps)

    v = Viewer(a)
    print(f"loaded {v.ck_info}")
    print(f"history: {v.stack} frames = {(v.stack - 1) * 40} ms lookback "
          f"@ 25 Hz, exec {EV.EXEC} x 40 ms")

    import socketserver
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state":
                with v.lock:
                    img, info = v.png, dict(v.info)
                info["img"] = img
                self._send(200, json.dumps(info).encode(), "application/json")
            elif self.path == "/health":
                self._send(200, json.dumps(dict(ok=True, ep=v.ep_i)).encode(),
                           "application/json")
            elif self.path == "/restart":
                v.restart.set()          # episode loop aborts at the next step
                self._send(200, json.dumps(dict(ok=True)).encode(),
                           "application/json")
            else:
                self._send(404, b"nope", "text/plain")

        def log_message(self, *a):
            pass

    class S(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = S(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"\n  serving {url}\n  (Ctrl-C to stop)\n", flush=True)
    if not a.no_open:
        try:
            import webbrowser
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    # THREADING: MuJoCo's classic Renderer binds its WGL context to the thread
    # that created it, so the rollout MUST run on THIS thread (the one that
    # built Viewer).  Rendering from a worker thread fails with
    # "WGL: Failed to make context current".  The HTTP handlers only ever read
    # v.png / v.info under v.lock, never the sim, so serving from daemon
    # threads is safe.
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        v.run()                       # <- main thread, owns the GL context
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        v.done.set()
        time.sleep(0.3)
        srv.shutdown()
        v.close()


if __name__ == "__main__":
    main()
