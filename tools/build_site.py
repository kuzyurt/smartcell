"""Assemble the static site for Cloudflare Pages into dist/.

    python tools/build_site.py          # then publish the dist/ folder

The static copy has no simulation server; the Play section falls back to a
recorded run and the commands to start the real thing locally.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
MEDIA = ["hero.mp4", "hero_poster.jpg", "before.mp4", "after.mp4", "policy.mp4", "policy_poster.jpg",
         "smoothing.json", "architecture.svg"]

HEADERS = """/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
/media/*
  Cache-Control: public, max-age=86400
"""

shutil.copytree(os.path.join(ROOT, "web", "static"), DIST, dirs_exist_ok=True)
os.makedirs(os.path.join(DIST, "media"), exist_ok=True)
for f in MEDIA:
    shutil.copy2(os.path.join(ROOT, "media", f), os.path.join(DIST, "media", f))
with open(os.path.join(DIST, "_headers"), "w") as fh:
    fh.write(HEADERS)
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DIST) for f in fs)
print(f"dist/ ready: {size / 1e6:.1f} MB")
