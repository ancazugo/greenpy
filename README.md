# greenpy

Measure the **3-30-300 rule** of urban greening for any city, powered by [Apache Sedona](https://sedona.apache.org/).

The 3-30-300 rule ([Konijnendijk, 2023](https://doi.org/10.1007/s11676-022-01523-z)) states that every home should:

1. **See at least 3 trees** — measured here as the number of trees within a buffer of each building (**T3**),
2. **Sit in a neighbourhood with at least 30 % canopy cover** (**T30**),
3. **Be within 300 m of a public green space** (**T300**).

greenpy computes each metric per census unit from standard geospatial inputs and merges them into a single analysis-ready table.

## Modules

| Process | What it computes | Output (one CSV per geo code) |
|---|---|---|
| `T3` | Trees (above a height/area threshold) within `--buffer` metres of each building | `building_id`, `tree_count_<buffer>m`, sub-geo code |
| `T30` | Canopy cover % per sub-geo unit, from CHM raster tiles, a GEE canopy-height asset, or tree polygons | sub-geo code, `canopy_cover`, `total_pixels` |
| `T300` | Road-network and Euclidean distance from each building to the nearest park | `building_id`, distances, closest park ids, sub-geo code |
| `Tree_count` | Total tree count per sub-geo unit (no size filter) | sub-geo code, `tree_count` |
| `Spectral` *(optional)* | Median spectral indices (NDVI, NDWI, …) per sub-geo unit via Google Earth Engine | sub-geo code, one column per index |
| `Merge` | Consolidates all module outputs into `database/T3_30_300_spectral.parquet` | one row per geo unit |

## Requirements

- Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/)
- A JDK (for Spark/Sedona), pointed to by `JDK_HOME`
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

## Input data

All inputs are vector files readable by GeoPandas (GeoPackage, Shapefile, GeoJSON, (Geo)Parquet…):

- **Buildings** — footprint polygons with a unique id column
- **Trees** — canopy polygons with height and area attributes (a single file or a directory of tiles)
- **Canopy for T30** — one of: a directory of CHM raster tiles (`.tif`, `chm_tiles_dir`); a GEE canopy-height asset (`canopy_height_ee_path`, needs `gee_project`); or the tree polygons above. See *Canopy cover source* below
- **Parks** — green-space polygons, plus access points (can be the same file)
- **Roads** — edges (and optionally nodes; nodes are derived from edge endpoints if absent)
- **Census boundaries** — one polygon per unit of the *finest* geography, with a column for every level of the hierarchy (e.g. district and tract codes)

On the first run, greenpy converts everything to a parquet cache in `<output.base_dir>/database/`, renaming your columns to canonical names. Delete that folder to rebuild the cache after changing input data or column mappings.

## Configuration

Each study area is described by a YAML config — see [`examples/generic.yaml`](examples/generic.yaml) for a fully commented template (plus `examples/westminster.yaml`, `examples/anglesey.yaml`, `examples/england.yaml` for UK setups). The key sections:

```yaml
study_area_name: MyCity
crs: EPSG:32632          # projected CRS in metres

data:                    # paths to the inputs above
  buildings: /path/to/buildings.gpkg
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
greenpy run -c config.yaml -p T300
greenpy run -c config.yaml -p Tree_count
greenpy run -c config.yaml -p Spectral        # optional, needs GEE
greenpy run -c config.yaml -p Merge
```

Useful options:

- `--geo_level` / `--sub_geo_level` — which levels of `columns.geo_levels` to iterate over / aggregate to (defaults: coarsest / finest)
- `--geo_code CODE` — process a single geography instead of all
- `--parallel --n_workers 4` — process geo codes concurrently (per-geo Spark views are isolated, so results match sequential runs)
- `--no-overwrite` — skip geo codes whose output CSV already exists (resume an interrupted run)
- `--query_method sql|rdd` — Sedona join strategy for T3 (default `rdd`)
- `--tree_area` / `--tree_height` — minimum canopy area (m²) and height (m) for a tree to count in T3
- `--low_threshold` / `--high_threshold` — canopy-height band in metres for T30 binarisation (default 3–60)
- `--gee_scale` — download resolution in metres for the T30 GEE canopy source (default `1.0`; raise to e.g. `10` for faster, lighter downloads over large regions)

### Canopy cover source (T30)

T30 picks its canopy source by what's configured, in priority order:

1. **CHM raster tiles** (`chm_tiles_dir`) — local `.tif` tiles, binarised to a canopy/no-canopy mask between `--low_threshold` and `--high_threshold` metres.
2. **GEE canopy-height asset** (`canopy_height_ee_path`) — binarised *server-side* in Google Earth Engine and downloaded with [xee](https://github.com/google/Xee); no local canopy data required. Needs `gee_project` in the config. Example asset (global 1 m Meta/WRI canopy height):

   ```yaml
   data:
     canopy_height_ee_path: projects/sat-io/open-datasets/facebook/meta-canopy-height
   gee_project: my-gee-project
   ```

   Downloaded rasters are cached under `<base_dir>/database/gee_canopy/<geo_code>.tif`. At the native 1 m scale large regions can be slow — use `--gee_scale 10` to trade detail for speed.
3. **Tree polygons** (`trees_dir`) — canopy cover as Σ tree area / unit area; used when no raster or GEE source is set.

`Merge` accepts `--t3_buffers` (default `10 25 50 75 100`) and combines whichever T3 buffer runs exist; Spectral output is included only if present.

## Output layout

```
<output.base_dir>/
├── T3/            T3_<code>_<buffer>m.csv
├── T30/           T30_<code>.csv
├── T300/          T300_<code>.csv
├── Tree_count/    Tree_count_<code>.csv
├── Spectral/      Spectral_<code>.csv
└── database/      parquet cache + consolidated outputs
    └── T3_30_300_spectral.parquet   ← final merged table
```
