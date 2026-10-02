"""Build a nested census-boundaries file for greenpy from geoBoundaries (gbOpen).

Downloads ADM1..ADMn for a country, keeps the finest units inside one ADM1
unit, and gives each finest unit the codes and names of its parents (taken
from the parent containing the unit's representative point, so the
hierarchy always nests). Output columns: ADM<k>_code / ADM<k>_name per level
plus geometry, ready for columns.geo_levels.

Usage:
    python scripts/prepare_geoboundaries.py --iso KEN --adm1 Nairobi --levels 3 \\
        --out /scratch/acz25/greenpy_runs/nairobi/input/nairobi_wards.parquet
"""

import argparse
import json
import urllib.request
from pathlib import Path

import geopandas as gpd

API = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/ADM{level}/"


def fetch_level(iso: str, level: int) -> gpd.GeoDataFrame:
    with urllib.request.urlopen(API.format(iso=iso, level=level), timeout=120) as r:
        meta = json.load(r)
    gdf = gpd.read_file(meta["gjDownloadURL"])
    gdf = gdf.rename(columns={"shapeID": f"ADM{level}_code", "shapeName": f"ADM{level}_name"})
    print(f"ADM{level}: {len(gdf)} units ({meta.get('boundarySource')}, {meta.get('boundaryYearRepresented')})")
    return gdf[[f"ADM{level}_code", f"ADM{level}_name", "geometry"]].to_crs(4326)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", required=True, help="ISO3 country code, e.g. KEN")
    ap.add_argument("--adm1", required=True, help="ADM1 unit name to keep, e.g. Nairobi")
    ap.add_argument("--levels", type=int, default=3, help="Finest ADM level to include")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    levels = {k: fetch_level(args.iso, k) for k in range(1, args.levels + 1)}
    adm1 = levels[1][levels[1]["ADM1_name"].str.lower() == args.adm1.lower()]
    if adm1.empty:
        raise SystemExit(f"No ADM1 unit named {args.adm1!r}; options: {sorted(levels[1]['ADM1_name'])}")

    finest = levels[args.levels].copy()
    points = finest.set_geometry(finest.representative_point())
    for k in range(1, args.levels):
        parents = adm1 if k == 1 else levels[k]
        joined = gpd.sjoin(points, parents, predicate="within", how="left").drop(columns="index_right")
        joined = joined[~joined.index.duplicated()]
        finest[f"ADM{k}_code"] = joined[f"ADM{k}_code"]
        finest[f"ADM{k}_name"] = joined[f"ADM{k}_name"]
    finest = finest[finest["ADM1_code"].isin(adm1["ADM1_code"])]

    cols = [c for k in range(1, args.levels + 1) for c in (f"ADM{k}_code", f"ADM{k}_name")]
    finest = finest[cols + ["geometry"]].reset_index(drop=True)
    missing = finest[cols].isna().any(axis=1).sum()
    area_units = finest.to_crs(finest.estimate_utm_crs()).union_all().area / 1e6
    area_adm1 = adm1.to_crs(finest.estimate_utm_crs()).union_all().area / 1e6
    print(f"{args.adm1}: {len(finest)} ADM{args.levels} units, "
          + ", ".join(f"{finest[f'ADM{k}_code'].nunique()} ADM{k}" for k in range(1, args.levels))
          + f"; units cover {area_units:.1f} of {area_adm1:.1f} km2; {missing} without a parent")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    finest.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
