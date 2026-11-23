"""Regenerate the README figures: media/architecture.svg, pipeline.svg, results.svg.

    python tools/make_figures.py

Plain SVG, transparent background, one accent colour, light and dark via
prefers-color-scheme (works when the file is shown through <img> on GitHub).
"""
import math
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")

STYLE = """<style>
:root{--ink:#1f2328;--line:#d1d9e0;--acc:#0b7a75;--on:#ffffff}
@media (prefers-color-scheme:dark){:root{--ink:#e6edf3;--line:#3d444d;--acc:#3dbdb2;--on:#0d1117}}
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;fill:var(--ink);font-size:13px}
.h{font-size:16px;font-weight:650}
.l{font-size:12.5px}
.t{font-size:12px;opacity:.7}
.box{fill:none;stroke:var(--ink);stroke-width:1.25}
.fill{fill:var(--acc);stroke:none}
.on{fill:var(--on)}
.ar{fill:none;stroke:var(--ink);stroke-width:1.5}
.hair{stroke:var(--line);stroke-width:1;fill:none}
.head{fill:var(--ink)}
.ptp{stroke:var(--ink);stroke-width:2;fill:none}
.pl{fill:var(--acc)}
.ph{fill:var(--ink)}
.dot{fill:var(--acc)}
.cic{fill:var(--acc)}
</style>
<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="head" d="M1 1.5L8.5 5L1 8.5z"/></marker></defs>"""


def svg(w, h, title, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{title}">\n<title>{title}</title>\n{STYLE}\n{body}\n</svg>\n')


def architecture():
    xs, w, y0, y1 = [20, 310, 600, 890], 190, 40, 120
    names = [("Policy", "learned, frozen"), ("Visual servo", "geometry, no learning"),
             ("Release primitive", "rules on joint state"), ("SO-101", "6 joints + gripper")]
    arrows = [("arm + grip", "target"), ("corrected", "target"), ("arm + grip", "command")]
    b = []
    for i, (x, (n, s)) in enumerate(zip(xs, names)):
        last = i == 3
        b.append(f'<rect class="{"fill" if last else "box"}" x="{x}" y="{y0}" width="{w}" height="{y1-y0}" rx="8"/>')
        cls = " on" if last else ""
        b.append(f'<text class="h{cls}" x="{x+w/2}" y="{y0+36}" text-anchor="middle">{n}</text>')
        b.append(f'<text class="{cls.strip()}" x="{x+w/2}" y="{y0+58}" text-anchor="middle">{s}</text>')
    for i, (a, c) in enumerate(arrows):
        x1, x2 = xs[i] + w, xs[i + 1]
        b.append(f'<path class="ar" d="M{x1+3} 80H{x2-4}" marker-end="url(#a)"/>')
        cx = (x1 + x2) / 2
        b.append(f'<text class="l" x="{cx}" y="56" text-anchor="middle">{a}</text>')
        b.append(f'<text class="l" x="{cx}" y="71" text-anchor="middle">{c}</text>')
    # sensors
    b.append('<rect class="box" x="20" y="232" width="770" height="56" rx="8"/>')
    b.append('<text class="h" x="40" y="266">Sensors</text>')
    b.append('<text x="128" y="265">wrist camera  ·  joint angles, measured and commanded</text>')
    for cx, lab in ((115, "8 gray frames + joints"), (405, "RGB frame"), (695, "gripper error, lift height")):
        b.append(f'<path class="ar" d="M{cx} 229V124" marker-end="url(#a)"/>')
        b.append(f'<text class="l" x="{cx+10}" y="182">{lab}</text>')
    b.append('<path class="ar" d="M985 124V260H794" marker-end="url(#a)"/>')
    b.append('<text class="l" x="803" y="250">25 Hz</text>')
    return svg(1100, 312, "Control pipeline: policy, visual servo and release primitive in series, "
               "each reading sensors, driving the SO-101", "\n".join(b))


def pipeline():
    xs = [20, 300, 580, 860]
    steps = [("Expert", ["243 demonstrations"]), ("Train", ["behaviour cloning,", "549,436 samples"]),
             ("DAgger", ["948 corrections from", "states the policy visits"]),
             ("Freeze and test", ["600 trials on", "unseen seeds"])]
    b = []
    for i, (x, (n, ls)) in enumerate(zip(xs, steps)):
        b.append(f'<circle class="cic" cx="{x+12}" cy="40" r="12"/>')
        b.append(f'<text class="h on" x="{x+12}" y="45.5" text-anchor="middle" style="font-size:13px">{i+1}</text>')
        b.append(f'<text class="h" x="{x+34}" y="45">{n}</text>')
        for k, t in enumerate(ls):
            b.append(f'<text x="{x+34}" y="{72+k*18}">{t}</text>')
        if i < 3:
            b.append(f'<path class="ar" d="M{x+232} 40H{xs[i+1]-8}" marker-end="url(#a)"/>')
    b.append('<path class="ar" d="M652 112V140H386V118" marker-end="url(#a)"/>')
    b.append('<text class="l" x="519" y="162" text-anchor="middle">corrections go back into training</text>')
    return svg(1100, 175, "Training loop: expert, train, DAgger corrections feeding back, then freeze and test", "\n".join(b))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def results():
    rows = [
        ("Stock cell", None),
        ("nominal", (199, 198, 200)), ("lighting, tint, mount jitter", (198, 195, 200)),
        ("held-out tote corners", (199, 199, 200)),
        ("Blender hall, no retraining", None),
        ("nominal", (100, 99, 100)), ("lighting, tint, mount jitter", (50, 50, 50)),
        ("held-out tote corners", (50, 50, 50)), ("visual servo switched off", (43, 30, 50)),
    ]
    X0, X1, LO, HI = 300, 800, 40.0, 100.0
    sx = lambda v: X0 + (v - LO) / (HI - LO) * (X1 - X0)
    top, rh, gh = 64, 34, 44
    b = []
    # column heads + direct labels
    b.append(f'<text class="l" x="880" y="34" text-anchor="end" style="font-weight:600">Pick</text>')
    b.append(f'<text class="l" x="960" y="34" text-anchor="end" style="font-weight:600">Place</text>')
    b.append(f'<text class="l" x="1040" y="34" text-anchor="end" style="font-weight:600">Trials</text>')
    y = top
    ys = []
    for lab, d in rows:
        if d is None:
            y += 14 if ys else 0
            b.append(f'<text class="h" x="20" y="{y+8}" style="font-size:14px">{lab}</text>')
            y += 28
            continue
        ys.append(y)
        y += rh
    plot_top, plot_bot = top - 6, y + 6
    for v in (50, 75, 100):
        b.append(f'<path class="hair" d="M{sx(v)} {plot_top}V{plot_bot}"/>')
        b.append(f'<text class="t" x="{sx(v)}" y="{plot_bot+20}" text-anchor="middle">{v} %</text>')
    y = top
    first = True
    for lab, d in rows:
        if d is None:
            y += 14 if y != top else 0
            y += 28
            continue
        k_pick, k_place, n = d
        yc = y + rh / 2 - 2
        b.append(f'<text x="38" y="{yc+4.5}">{lab}</text>')
        for kk, cls, dy in ((k_pick, "ph", -5), (k_place, "pl", 5)):
            lo, hi = wilson(kk, n)
            yy = yc + dy
            b.append(f'<path class="ptp" d="M{sx(max(LO, lo*100)):.1f} {yy}H{sx(hi*100):.1f}" style="stroke:{"var(--ink)" if cls=="ph" else "var(--acc)"};stroke-width:1.5"/>')
            if cls == "ph":
                b.append(f'<circle cx="{sx(kk/n*100):.1f}" cy="{yy}" r="4.5" style="fill:none;stroke:var(--ink);stroke-width:1.5"/>')
            else:
                b.append(f'<circle class="dot" cx="{sx(kk/n*100):.1f}" cy="{yy}" r="4.5"/>')
        b.append(f'<text x="880" y="{yc+4.5}" text-anchor="end">{k_pick/n*100:.1f}</text>')
        b.append(f'<text x="960" y="{yc+4.5}" text-anchor="end" style="font-weight:650;fill:var(--acc)">{k_place/n*100:.1f}</text>')
        b.append(f'<text x="1040" y="{yc+4.5}" text-anchor="end">{n}</text>')
        y += rh
    # direct series labels on the first data row
    fy = ys[0] + rh / 2 - 2
    b.append(f'<text class="l" x="{sx(98.4)+14:.0f}" y="{fy-8}" style="fill:var(--ink)">pick</text>') if False else None
    b.append(f'<text class="l" x="{X0}" y="34" style="fill:var(--ink)"><tspan>○ pick  </tspan><tspan style="fill:var(--acc);font-weight:650">● place</tspan><tspan>   with 95 % Wilson interval</tspan></text>')
    h = int(plot_bot + 40)
    return svg(1080, h, "Pick and place rates with 95 percent Wilson intervals for the stock cell and the "
               "Blender-dressed hall; switching the visual servo off drops placement to 60 percent", "\n".join(b))


if __name__ == "__main__":
    for name, fn in (("architecture", architecture), ("pipeline", pipeline), ("results", results)):
        with open(os.path.join(OUT, name + ".svg"), "w") as f:
            f.write(fn())
        print("wrote", name)
    for k, n in ((99, 100), (198, 200), (30, 50), (50, 50)):
        lo, hi = wilson(k, n)
        print(f"check wilson {k}/{n}: {lo*100:.1f}-{hi*100:.1f}")
