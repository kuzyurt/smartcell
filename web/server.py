"""smartcell web demo: ``python web/server.py`` (or ``python -m web``).

Serves the project page and a live, controllable simulation. The simulation
runs the frozen policy on this machine; the browser only displays frames and
sends settings.
"""
import argparse
import json
import mimetypes
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
STATIC = os.path.join(HERE, "static")


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _stream(self, which):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    img, seq = engine.jpeg(which)
                    if img is None or seq == seen:
                        time.sleep(0.015)
                        continue
                    seen = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(img)).encode() + b"\r\n\r\n" + img + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/state":
                return self._json(engine.get_state())
            if path == "/stream/main.mjpg":
                return self._stream("main")
            if path == "/stream/wrist.mjpg":
                return self._stream("wrist")
            if path == "/":
                path = "/index.html"
            base = STATIC
            if path.startswith("/media/"):          # shared with the README
                base, path = os.path.join(ROOT, "media"), path[len("/media"):]
            full = os.path.normpath(os.path.join(base, path.lstrip("/")))
            if not full.startswith(base) or not os.path.isfile(full):
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            with open(full, "rb") as fh:
                data = fh.read()
            return self._send(200, data, ctype, {"Cache-Control": "no-cache"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            if self.path == "/api/config":
                engine.set_cfg(body)
            elif self.path == "/api/control":
                engine.control(str(body.get("action", "")))
            else:
                return self._json({"error": "unknown"}, 404)
            return self._json(engine.get_state())

        def log_message(self, *a):
            pass

    return H


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--autostart", action="store_true")
    a = ap.parse_args()

    from web.engine import Engine
    eng = Engine(ckpt=a.ckpt)
    eng._cell(eng.cfg["scene"])                  # build the default scene up front
    if a.autostart:
        eng.control("start")
    srv = Server((a.host, a.port), make_handler(eng))
    url = f"http://{a.host}:{a.port}/"
    print(f"\n  smartcell is running at {url}   (device: {eng.dev}, Ctrl-C to stop)\n",
          flush=True)
    if not a.no_open:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        eng.run_forever()                        # main thread owns the GL context
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        eng.run["shutdown"] = True
        srv.shutdown()


if __name__ == "__main__":
    main()
