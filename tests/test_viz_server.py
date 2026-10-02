import gzip
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

pytest.importorskip("duckdb")

from greenpy.viz.catalog import build_catalog
from greenpy.viz.server import make_handler
from greenpy.viz.store import connect, ensure_store, read_meta
from greenpy.viz.tiles import TileStore

from test_viz_store import _tile_xy
from viz_fixtures import make_config, make_outputs, make_trees


@pytest.fixture
def server(tmp_path):
    make_outputs(tmp_path, merged=True)
    cat = build_catalog(make_config(tmp_path, make_trees(tmp_path / "trees.gpkg")))
    path = ensure_store(cat)
    con = connect(path, read_only=True)
    meta = read_meta(con)
    con.close()
    store = TileStore(path, meta)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    store.close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers, r.read()


def test_catalog_and_page(server):
    status, _, body = _get(server + "/api/catalog")
    cat = json.loads(body)
    assert status == 200 and cat["study_area_name"] == "Testville"
    assert cat["min_zoom"]["buildings"] == 14 and cat["trees"]["count"] == 3
    status, headers, body = _get(server + "/")
    assert status == 200 and b"maplibre-gl.js" in body and headers["Content-Type"] == "text/html"
    assert _get(server + "/static/app.js")[0] == 200


def test_tiles_stats_feature(server):
    x, y = _tile_xy(16)
    status, headers, body = _get(f"{server}/tiles/buildings/tree_count_50m/16/{x}/{y}.pbf")
    assert status == 200 and headers["Content-Encoding"] == "gzip" and gzip.decompress(body)
    assert _get(f"{server}/tiles/buildings/tree_count_50m/16/{x + 40}/{y}.pbf")[0] == 204
    assert _get(f"{server}/tiles/trees/_/16/{x}/{y}.pbf")[0] == 200
    stats = json.loads(_get(server + "/api/stats/buildings/meets_3")[2])
    assert stats["true"] == 2
    feat = json.loads(_get(server + "/api/feature/TRACT/T1")[2])
    assert feat["canopy_cover"] == 41.0


@pytest.mark.parametrize("path", [
    "/tiles/buildings/not_a_metric/16/0/0.pbf",
    "/tiles/nope/x/16/0/0.pbf",
    "/api/feature/buildings/999",
    "/api/stats/buildings/nope",
    "/static/../../store.py",
])
def test_not_found(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server + path)
    assert e.value.code == 404
