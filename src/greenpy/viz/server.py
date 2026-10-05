"""Local HTTP server for the dashboard: static page, JSON API and vector tiles.

Routes:
    /                                 index.html
    /static/<file>                    front-end assets
    /api/catalog                      layers, metrics, bounds, tree availability
    /api/stats/<layer>/<metric>       legend stats (breaks, histogram)
    /api/feature/<layer>/<id>         all attributes of one building or unit
    /tiles/<layer>/<metric>/<z>/<x>/<y>.pbf   (metric "_" for none, e.g. trees)
"""

import json
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from loguru import logger

from .tiles import MIN_ZOOM, TileStore

STATIC_DIR = Path(__file__).parent / "static"
_TILE = re.compile(r"^/tiles/([^/]+)/([^/]+)/(\d+)/(\d+)/(\d+)\.pbf$")
_STATS = re.compile(r"^/api/stats/([^/]+)/([^/]+)$")
_FEATURE = re.compile(r"^/api/feature/([^/]+)/(.+)$")


def _catalog_payload(store: TileStore) -> dict:
    meta = store.meta
    return {
        "study_area_name": meta["study_area_name"],
        "bounds": meta["bounds"],
        "layers": {
            name: {"label": info["label"], "metrics": info["metrics"]}
            for name, info in meta["layers"].items()
        },
        "trees": meta.get("trees"),
        "parks": meta.get("parks"),
        "summary": meta.get("summary"),
        "min_zoom": MIN_ZOOM,
    }


def make_handler(store: TileStore) -> type[BaseHTTPRequestHandler]:
    catalog = json.dumps(_catalog_payload(store), default=str).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "greenpy-viz"

        def log_message(self, fmt, *args):  # route access logs through loguru at debug level
            logger.debug("viz: " + fmt % args)

        def _send(self, status: int, body: bytes = b"", ctype: str = "application/json", extra: dict | None = None):
            self.send_response(status)
            if body:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj) -> None:
            body = obj.encode() if isinstance(obj, str) else json.dumps(obj, default=str).encode()
            self._send(200, body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            try:
                if m := _TILE.match(path):
                    self._tile(*m.groups())
                elif path == "/api/catalog":
                    self._send(200, catalog)
                elif m := _STATS.match(path):
                    stats = store.stats(*m.groups())
                    self._json(stats) if stats is not None else self._send(404)
                elif m := _FEATURE.match(path):
                    layer, fid = m.groups()
                    if layer not in store.meta["layers"]:
                        return self._send(404)
                    feature = store.feature(layer, fid)
                    self._json(feature) if feature is not None else self._send(404)
                else:
                    self._static(path)
            except KeyError:
                self._send(404)
            except BrokenPipeError:
                pass
            except Exception as e:  # keep serving other requests
                logger.exception(f"viz: error serving {path}: {e}")
                self._send(500)

        def _tile(self, layer, metric, z, x, y):
            if not store.has_layer(layer):
                return self._send(404)
            data = store.tile(layer, None if metric == "_" else metric, int(z), int(x), int(y))
            if data is None:
                return self._send(204)
            self._send(200, data, "application/vnd.mapbox-vector-tile", {"Content-Encoding": "gzip"})

        def _static(self, path):
            rel = "index.html" if path in ("", "/") else path.removeprefix("/static/")
            file = (STATIC_DIR / rel).resolve()
            if not file.is_relative_to(STATIC_DIR.resolve()) or not file.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            self._send(200, file.read_bytes(), ctype, {"Cache-Control": "no-cache"})

    return Handler


def serve(store: TileStore, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Serve the dashboard until interrupted."""
    httpd = ThreadingHTTPServer((host, port), make_handler(store))
    httpd.daemon_threads = True
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{httpd.server_address[1]}/"
    logger.info(f"greenpy viz running at {url} (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("viz: stopped")
    finally:
        httpd.server_close()
        store.close()
