# Deploying the site (Cloudflare Pages)

The page in `web/static/` works in two modes. Started locally with
`python web/server.py` it controls a live simulation. Hosted as static files
it shows a recorded run and the commands to start the real thing, because
Cloudflare Pages cannot run the simulation.

## Connect the repository

1. Cloudflare dashboard, Workers & Pages, Create, Pages, Connect to Git.
2. Pick this repository and the branch to publish.
3. Build settings:

| Setting | Value |
|---|---|
| Framework preset | None |
| Build command | `python tools/build_site.py` |
| Build output directory | `dist` |

The build copies `web/static/` and the few files it needs from `media/`
(about 19.4 MB) into `dist/` and adds a `_headers` file. It uses only the
Python standard library, so no install step is needed.

## Or upload by hand

```bash
python tools/build_site.py
npx wrangler pages deploy dist --project-name smartcell
```

## Notes

- Individual files must stay under 25 MiB on Pages; the largest is `hero.mp4`.
- The language switch (EN/DE) and all content are client side.
- Without a `404.html`, Cloudflare Pages serves `index.html` (`HTTP 200`,
  `text/html`) for `/api/state` as its SPA fallback. `app.js` detects the
  non-JSON response and switches to static recorded-video mode.
