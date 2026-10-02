# greenpy

Measure the **3-30-300 rule** of urban greening for any city, powered by [Apache Sedona](https://sedona.apache.org/).

The 3-30-300 rule ([Konijnendijk, 2023](https://doi.org/10.1007/s11676-022-01523-z)) states that every home should:

1. **See at least 3 trees** — measured here as the number of trees within a buffer of each building (**T3**),
2. **Sit in a neighbourhood with at least 30 % canopy cover** (**T30**),
3. **Be within 300 m of a public green space** (**T300**).

greenpy computes each metric per building and per census unit from standard geospatial inputs, evaluates the rule for every building, and merges everything into a single analysis-ready table.

## Modules

| Process | What it computes | Output (one CSV per geo code) |
|---|---|---|
| `T3` | Trees (height and area strictly above `--tree_height`/`--tree_area`) whose centroid lies within `--buffer` metres of each building | `building_id`, `tree_count_<buffer>m`, sub-geo code |
| `T30` | Canopy cover % per sub-geo unit, from CHM raster tiles, a GEE canopy-height asset, or tree polygons | sub-geo code, `canopy_cover`, `total_pixels` |
| `T30_buildings` | Canopy cover % within `--buffer` metres of each building (distributed Sedona `RS_ZonalStats` for raster sources), same canopy sources as T30 | `building_id`, `tree_pixels`, `total_pixels`, `canopy_cover` |
| `T300` | Road-network and Euclidean distance from each building to the nearest park | `building_id`, distances, closest park ids, sub-geo code |
| `Tree_count` | Total tree count per sub-geo unit, each tree counted once in the unit containing its centroid (no size filter, unlike T3) | sub-geo code, `tree_count` |
| `Visibility` *(optional)* | Trees actually visible from each building via 2.5D line-of-sight, accounting for obstruction by other buildings and trees | `building_id`, `visible_trees_bottom/_middle/_top`, `visible_trees`, sub-geo code |
| `Spectral` *(optional)* | Spectral indices (NDVI, NDWI, …) per sub-geo unit via Google Earth Engine: a per-pixel temporal composite over the date range (`--composite max`, the default, or `median`), then the spatial median over each unit | sub-geo code, one column per index |
| `Merge` | Evaluates the 3-30-300 rule per building and consolidates all module outputs into `database/T3_30_300_spectral.parquet` | one row per geo unit (see *Merge* below) |

## Requirements

- Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/)
- A JDK (for Spark/Sedona), pointed to by `JDK_HOME` — Java 11–17 (Sedona ≥ 1.9 jars are built for Java 11+; Spark 3.5 supports up to 17)
- For `Spectral` only: a Google Earth Engine project and a boundaries asset uploaded to GEE

## Installation

```bash
uv sync
source /maps/acz25/envs/greenpy-env/bin/activate   # project environment
```

> The environment lives outside the repo, so installing new dependencies needs the `--active` flag: `uv add --active <package>`.

Create a `.env` file in the repo root:

```bash
DATA_DIR=/path/to/your/data
JDK_HOME=/path/to/jdk            # used to set JAVA_HOME for Spark
GEE_PROJECT_NAME=my-gee-project  # only needed for Spectral
```

## Run with Docker

If you can't install a compatible JDK natively, use the Docker image — it bundles OpenJDK 17, Python 3.12, and the Sedona JARs (pre-downloaded at build time, so the container runs offline).

```bash
docker build -t greenpy .
```

Mount your data and a writable output dir, and reference the **container** paths in your YAML config (e.g. `data.buildings: /data/buildings.gpkg`, `output.base_dir: /work/output`):

```bash
docker run --rm \
  -v /path/to/data:/data -e DATA_DIR=/data \
  -v "$PWD/output":/work \
  greenpy run -c /data/config.yaml -p T3 --buffer 100
```

The `greenpy` entrypoint is baked in, so pass only the CLI arguments. Outputs, `logs/`, `spark-warehouse/` and `cache/` are written under `/work`.

**Spark memory.** The image defaults to a laptop-friendly `SPARK_DRIVER_MEMORY=4g` / `SPARK_EXECUTOR_MEMORY=2g`. Raise them on a bigger host:

```bash
docker run --rm -e SPARK_DRIVER_MEMORY=32g -e SPARK_EXECUTOR_MEMORY=16g ... greenpy run ...
```

(These env vars also work for native runs; unset, they keep the original 64g/32g defaults.)

**docker compose.** [`docker-compose.yml`](docker-compose.yml) wires up the `./data`, `./config` and `./output` mounts:

```bash
docker compose run --rm greenpy run -c /config/city.yaml -p T3 --buffer 100
```

**GEE features** (`Spectral`, GEE canopy, `open_buildings`) need Google Earth Engine, whose login is interactive. Authenticate once on the host (`earthengine authenticate`), then mount the credentials and set the project — see the commented lines in `docker-compose.yml`, or add to `docker run`:

```bash
-v "$HOME/.config/earthengine":/home/greenpy/.config/earthengine:ro \
-e GEE_PROJECT_NAME=my-gee-project
```

The core 3-30-300 pipeline (T3, T30, T300, Tree_count, Visibility, Merge) with local data needs no GEE.

**Results map.** Publish the port and bind to all interfaces inside the container: `docker run --rm -p 8765:8765 ... greenpy viz -c /data/config.yaml --host 0.0.0.0 --no-browser`, then open http://localhost:8765.

## Input data

All inputs are vector files readable by GeoPandas (GeoPackage, Shapefile, GeoJSON, (Geo)Parquet…):

- **Buildings** — footprint polygons with a unique id column. Instead of a file, `data.buildings` also accepts `osm` (OpenStreetMap; no heights), `overture` (Overture Maps, global; `height` feeds `building_height` but is sparse outside major cities), or `open_buildings` (Google Open Buildings v3 via GEE, needs `gee_project`; no heights, and covers Africa, South/Southeast Asia and Latin America & the Caribbean only — **not** Europe or North America)
- **Trees** — canopy polygons with height and area attributes (a single file or a directory of tiles)
- **Canopy for T30** — one of: a directory of CHM raster tiles (`.tif`, `chm_tiles_dir`); a GEE canopy-height asset (`canopy_height_ee_path`, needs `gee_project`); or the tree polygons above. See *Canopy cover source* below
- **Parks** — green-space polygons, plus access points (can be the same file). Set `park_min_area_ha` (top-level config key) to ignore small green spaces — the WHO guideline behind the 300 rule uses ≥ 0.5–1 ha. Access points are only filtered along with their parks when `columns.park_access_ref_col` links them
- **Roads** — edges (and optionally nodes; nodes are derived from edge endpoints if absent). File-based networks must be **noded** (split at every intersection — nodes are derived from segment endpoints only) and reduced to their **largest connected component**, or buildings snapping to isolated fragments get null network distances. `columns.road_edge_length` should hold edge lengths in metres; if the column is missing, lengths are computed from the geometry in the config CRS (with a warning)
- **Census boundaries** — one polygon per unit of the *finest* geography, with a column for every level of the hierarchy (e.g. district and tract codes)

On the first run, greenpy converts everything to a parquet cache in `<output.base_dir>/database/`, renaming your columns to canonical names. Delete that folder to rebuild the cache after changing input data or column mappings.

## Configuration

Each study area is described by a YAML config — see [`examples/generic.yaml`](examples/generic.yaml) for a fully commented template (plus `examples/westminster.yaml`, `examples/anglesey.yaml`, `examples/england.yaml` for UK setups). The key sections:

```yaml
study_area_name: MyCity
crs: EPSG:32632          # projected CRS in metres

data:                    # paths to the inputs above
  buildings: /path/to/buildings.gpkg   # or "osm" / "overture" / "open_buildings"
  trees_dir: /path/to/trees/
  chm_tiles_dir: null    # alternative to trees_dir for T30
  ...

columns:                 # your column names → greenpy's canonical names
  building_id: building_id
  tree_height_col: height
  tree_area_col: area
  geo_levels:            # geography hierarchy, coarsest → finest
    - district_code
    - tract_code

output:
  base_dir: /path/to/output/
```

## Usage

Run one module at a time; `Merge` last:

```bash
greenpy run -c config.yaml -p T3 --buffer 100
greenpy run -c config.yaml -p T30
greenpy run -c config.yaml -p T30_buildings --buffer 100
greenpy run -c config.yaml -p T300
greenpy run -c config.yaml -p Tree_count
greenpy run -c config.yaml -p Visibility      # optional, needs building heights
greenpy run -c config.yaml -p Spectral        # optional, needs GEE
greenpy run -c config.yaml -p Merge
greenpy viz -c config.yaml                    # interactive map, see "Visualising results"
```

Useful options:

- `--geo_level` / `--sub_geo_level` — which levels of `columns.geo_levels` to iterate over / aggregate to (defaults: coarsest / finest); Merge's `--geo_level` defaults to the second-finest level instead
- `--geo_code CODE` — process a single geography instead of all
- `--parallel --n_workers 4` — process geo codes concurrently (per-geo Spark views are isolated, so results match sequential runs)
- `--no-overwrite` — skip geo codes whose output CSV already exists (resume an interrupted run)
- `--query_method sql|rdd` — Sedona join strategy for T3 (default `rdd`)
- `--tree_area` / `--tree_height` — a tree counts in T3 and Visibility only if its canopy area (m²) and height (m) are *strictly greater* than these (defaults 10 and 3)
- `--observer_mode facade|centroid` — where Visibility sightlines start on the building (default `facade`)
- `--low_threshold` / `--high_threshold` — canopy-height band in metres for T30/T30_buildings binarisation (default 3–60)
- `--gee_scale` — download resolution in metres for the T30/T30_buildings GEE canopy source (default `1.0`; raise to e.g. `10` for faster, lighter downloads over large regions — canopy is still thresholded at native resolution and each coarse pixel stores its canopy fraction, so cover stays unbiased)
- `--composite max|median` — Spectral temporal composite (default `max`)
- `--rule_t3_buffer`, `--rule_t30_buffer`, `--rule_distance` — how Merge evaluates the rule per building (see *Merge* below)

### Canopy cover source (T30, T30_buildings)

T30 and T30_buildings pick their canopy source by what's configured, in priority order:

1. **CHM raster tiles** (`chm_tiles_dir`) — local `.tif` tiles, binarised to a canopy/no-canopy mask between `--low_threshold` and `--high_threshold` metres.
2. **GEE canopy-height asset** (`canopy_height_ee_path`) — binarised *server-side* in Google Earth Engine and downloaded with [xee](https://github.com/google/Xee); no local canopy data required. Needs `gee_project` in the config. Example asset (global 1 m Meta/WRI canopy height):

   ```yaml
   data:
     canopy_height_ee_path: projects/sat-io/open-datasets/facebook/meta-canopy-height
   gee_project: my-gee-project
   ```

   Downloaded rasters are cached under `<base_dir>/database/gee_canopy/`, named by geo code, height band and scale. At the native 1 m scale large regions can be slow — use `--gee_scale 10` to trade detail for speed: the mask is binarised at the native 1 m and averaged to canopy fraction per 10 m pixel server-side, so cover estimates stay consistent with 1 m runs.
3. **Tree polygons** (`trees_dir`) — canopy cover as the area of the union of crowns clipped to each unit / unit area (a crown straddling two units counts only its own part in each; overlapping crowns are not double-counted); used when no raster or GEE source is set. Point trees fall back to their `tree_area` attribute.

Where T30 reports canopy per sub-geo unit, `T30_buildings` reports it per building, within `--buffer` metres of each footprint (`--buffer 0` for the footprint alone). Raster sources run distributed through Sedona (`RS_TileExplode` + `RS_ZonalStats`), so large CHM tile sets scale across Spark workers; the vector source uses an `ST_Intersection` area ratio. CHM nodata pixels are excluded from `total_pixels` in both modules. Two differences to keep in mind: raster tiles must not overlap (pixel counts are summed across tiles), and T30_buildings counts every pixel *touching* a buffer while T30 counts pixels whose *centre* falls in the unit. CHM tiles are searched recursively under `chm_tiles_dir`. In `Merge`, per-building canopy is averaged up to the geo level as `building_canopy_cover_<buffer>m` (included automatically when T30_buildings output exists).

### Tree visibility (Visibility, optional)

Where T3 counts trees *near* a building, `Visibility` checks whether they can actually be *seen* from it, using a 2.5D line-of-sight analysis: for every building–tree pair within `--buffer` metres, 9 sightlines are traced from three observer levels on the building (bottom z=0, middle z=H/2, top z=H) to three target levels on the tree (z=0, h/2, h). A sightline is blocked when another building footprint or tree canopy crosses it and that obstacle's height reaches the sightline's height at the crossing. A tree counts as visible from a level if at least one of its three target levels has a clear sightline.

Requirements and behaviour:

- **Building heights are required** — set `columns.building_height_col` (metres) in the config, then delete `<output.base_dir>/database/buildings.parquet` if the cache already exists. Tree heights come from `tree_height_col` as usual.
- **Complete height data is expected**: buildings or trees with a missing/invalid height are skipped entirely — as observers, targets *and* obstacles — with a warning, so gaps in height coverage bias the results. Not available with `buildings: osm` or `buildings: open_buildings` (those footprints carry no height); with `buildings: overture`, footprints without a height are dropped with a warning.
- `--observer_mode facade` (default) starts each sightline at the nearest point of the building footprint boundary to the tree (a window facing it); `centroid` uses the building centroid for all sightlines.
- Model assumptions: buildings are flat-topped prisms, trees are solid ground-to-crown prisms (a sightline under a canopy counts as blocked), terrain is flat, and grazing contact blocks.
- The obstruction join grows quickly with `--buffer` in dense areas — prefer modest buffers (e.g. 50–100 m).

### Merge and the 3-30-300 rule

`Merge` accepts `--t3_buffers` (default `10 25 50 75 100`) and combines whichever T3 buffer runs exist; Spectral, T30_buildings and Visibility outputs are included only if present. It aggregates to `--geo_level`, which for Merge defaults to the **second-finest** level of `columns.geo_levels` (the finest with a DGGS), reading sub-geo results at `--sub_geo_level` (default: finest).

Per unit it reports means — `tree_count_<b>m`, `canopy_cover` (area-weighted), `park_distance_manhattan` (road network) and `park_distance_euclidean`, plus `building_canopy_cover_<b>m` and `visible_trees_<b>m` when available — and `total_trees`.

Because the rule is a test every home should pass, Merge also evaluates it **per building** and writes `database/T3_30_300_buildings.parquet` with `meets_3`, `meets_30`, `meets_300` and `meets_3_30_300`. The per-unit table gains `pct_meets_3`, `pct_meets_30`, `pct_meets_300` and `pct_meets_3_30_300` (% of buildings passing):

- **3** — T3 count within `--rule_t3_buffer` metres (default `50`) is ≥ 3
- **30** — canopy cover of the building's sub-geo unit (the neighbourhood reading of the rule) is ≥ 30 %; with `--rule_t30_buffer N`, T30_buildings canopy within N metres of the building instead
- **300** — distance to the nearest park is ≤ 300 m, straight-line by default (`--rule_distance euclidean`, as in the WHO guideline) or `network`

A criterion with missing input is left null and excluded from that percentage; the combined rule is only evaluated where all three are known. T3 and Visibility counts are proxies for "seeing" trees: T3 counts trees near the building, Visibility checks line of sight.

### Per-building outputs

Each building belongs to exactly one census unit — the one containing its representative point — and is output only by that unit's run, including buildings that straddle unit boundaries. Buildings whose representative point lies outside every unit are skipped. T3's default `rdd` query path omits buildings with no tree in the buffer (the `sql` path reports them as 0); Merge fills them with 0. T300 keeps every building: with no reachable park within `osm.fetch_buffer` (default 2 km) its distances are null.

## Visualising results

`greenpy viz` opens an interactive map of whatever has been computed so far:

```bash
greenpy viz -c config.yaml            # opens http://localhost:8765
```

- **Buildings** are coloured by any per-building metric (T3 tree counts, park distances, T30_buildings canopy, visible trees and, after Merge, the `meets_*` rule flags). Choose quantile, equal-interval or **rule** classes; the last uses diverging colours centred on the 3-30-300 threshold (3 trees, 30 %, 300 m).
- **Units** (every census level, plus any DGGS grid already built) can be drawn as outlines over the buildings or filled with a unit metric: Merge's table at its level, T30/Tree_count/Spectral at the sub-geo level, and averages of the building metrics at every level.
- **Trees** from `data.trees_dir` are drawn to scale (crown radius from the area column or polygon area) or as dots when only points are available.
- Drag across the legend histogram to show only a value range; click a feature for all of its values.
- Basemaps: none, OpenFreeMap, CARTO, OpenStreetMap, or Sentinel-2 imagery (EOX, non-commercial).

Merge is optional: module CSVs are read directly when its parquets are missing. The first start builds `database/viz.duckdb` (geometry reprojected and indexed, statistics precomputed), and later starts reuse it until an output changes; `--rebuild` forces a rebuild and `--no-trees` skips the tree layer. Buildings and trees appear from zoom 14 (a view about 3.5 km across), which keeps whole-city datasets responsive; zoomed further out, show units instead.

The server listens on `127.0.0.1` only. On a remote machine, forward the port (`ssh -L 8765:localhost:8765 host`) and pass `--no-browser`. DuckDB downloads its spatial extension on first use; set `GREENPY_DUCKDB_EXTENSIONS=/some/dir` if your home directory is full or read-only.

## Output layout

```
<output.base_dir>/
├── T3/            T3_<code>_<buffer>m.csv
├── T30/           T30_<code>.csv
├── T30_buildings/ T30_buildings_<code>_<buffer>m.csv
├── T300/          T300_<code>.csv
├── Tree_count/    Tree_count_<code>.csv
├── Visibility/    Visibility_<code>_<buffer>m.csv
├── Spectral/      Spectral_<code>.csv
└── database/      parquet cache + consolidated outputs
    ├── T3_30_300_buildings.parquet  ← per-building 3-30-300 evaluation
    ├── T3_30_300_spectral.parquet   ← final merged table
    └── viz.duckdb                   ← map store built by `greenpy viz`
```
