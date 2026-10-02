"""A tiny greenpy output directory for the viz tests (British National Grid, central London)."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from greenpy.config.schema import ColumnMapping, DataPaths, GreenPyConfig, OutputPaths

CRS = "EPSG:27700"
X0, Y0 = 530000, 180000


def make_config(base: Path, trees: Path | None = None, **columns) -> GreenPyConfig:
    return GreenPyConfig(
        study_area_name="Testville",
        crs=CRS,
        data=DataPaths(
            buildings="unused", parks_sites="unused", parks_access="unused", roads="unused",
            census_boundaries="unused", trees_dir=str(trees) if trees else None,
        ),
        columns=ColumnMapping(building_id="building_id", geo_levels=["DIST", "TRACT"], **columns),
        output=OutputPaths(base_dir=str(base)),
        park_min_area_ha=0.5,
    )


def make_outputs(base: Path, merged: bool = False) -> None:
    """Buildings B0-B2 in tracts T0 (B0, B1) and T1 (B2), both in district D0.

    Module CSVs: T3 at 10 m (omitting B0) and 50 m, T300, T30 (per tract)
    and Tree_count (omitting T0).
    With merged=True also the Merge parquets T3_50m, T3_30_300_buildings and
    T3_30_300_spectral (at DIST), and parks P0 (1 ha, counted for 300) and P1
    (0.25 ha, below park_min_area_ha).
    """
    db = base / "database"
    db.mkdir(parents=True)
    buildings = gpd.GeoDataFrame(
        {"building_id": [0, 1, 2]},
        geometry=[shapely.box(X0 + 10 * i + 2, Y0 + 2, X0 + 10 * i + 8, Y0 + 8) if i < 2
                  else shapely.box(X0 + 150, Y0 + 2, X0 + 160, Y0 + 8) for i in range(3)],
        crs=CRS,
    )
    buildings.to_parquet(db / "buildings.parquet", index=False)
    census = gpd.GeoDataFrame(
        {"DIST": ["D0", "D0"], "TRACT": ["T0", "T1"], "TRACT_NAME": ["Northfield", "Southbank"]},
        geometry=[shapely.box(X0, Y0, X0 + 100, Y0 + 100), shapely.box(X0 + 100, Y0, X0 + 200, Y0 + 100)],
        crs=CRS,
    )
    census.to_parquet(db / "census_boundaries.parquet", index=False)
    gpd.GeoDataFrame(
        {"park_id": ["P0", "P1"], "name": ["Big Park", None]},
        geometry=[shapely.box(X0 + 20, Y0 + 20, X0 + 120, Y0 + 120), shapely.box(X0 + 150, Y0 + 20, X0 + 200, Y0 + 70)],
        crs=CRS,
    ).to_parquet(db / "parks_sites.parquet", index=False)
    pd.DataFrame({"building_id": [0, 1, 2], "DIST": ["D0"] * 3, "TRACT": ["T0", "T0", "T1"]}).to_parquet(
        db / "census_buildings_overlay.parquet", index=False
    )

    for d in ("T3", "T300", "T30", "Tree_count"):
        (base / d).mkdir()
    # like T3's rdd path, the 10 m run omits B0, which has no tree in its buffer
    for buf, ids, counts in ((10, [1, 2], [1, 4]), (50, [0, 1, 2], [2, 3, 9])):
        pd.DataFrame({"building_id": ids, f"tree_count_{buf}m": counts}).to_csv(
            base / "T3" / f"T3_D0_{buf}m.csv", index=False
        )
    pd.DataFrame({
        "building_id": [0, 1, 2], "closest_park_access_id": [7, 7, 8], "distance_manhattan": [120.0, 410.0, None],
        "closest_park_site_id": [1, 1, 2], "distance_euclidean": [100.0, 350.0, 80.0], "TRACT": ["T0", "T0", "T1"],
    }).to_csv(base / "T300" / "T300_D0.csv", index=False)
    pd.DataFrame({"TRACT": ["T0", "T1"], "canopy_cover": [12.5, 41.0], "total_pixels": [1e4, 1e4]}).to_csv(
        base / "T30" / "T30_D0.csv", index=False
    )
    # tree-less units are absent from Tree_count's CSVs
    pd.DataFrame({"TRACT": ["T1"], "tree_count": [13]}).to_csv(base / "Tree_count" / "Tree_count_D0.csv", index=False)

    if merged:
        pd.DataFrame({"building_id": [0, 1, 2], "tree_count_50m": [2, 3, 9]}).to_parquet(db / "T3_50m.parquet", index=False)
        pd.DataFrame({
            "building_id": [0, 1, 2], "DIST": ["D0"] * 3, "TRACT": ["T0", "T0", "T1"],
            # the values the rule tested, as Merge writes them beside the flags
            "tree_count_50m": [2, 3, 9], "distance_euclidean": [100.0, 350.0, 80.0], "canopy_cover": [12.5, 12.5, 41.0],
            "meets_3": [False, True, True], "meets_30": [False, False, True],
            "meets_300": [True, False, True], "meets_3_30_300": [False, False, True],
        }).astype({f: "boolean" for f in ("meets_3", "meets_30", "meets_300", "meets_3_30_300")}).to_parquet(
            db / "T3_30_300_buildings.parquet", index=False
        )
        pd.DataFrame({
            "DIST": ["D0"], "total_trees": [18], "tree_count_50m": [4.67], "canopy_cover": [26.75],
            "park_distance_euclidean": [176.67], "pct_meets_3_30_300": [33.3],
        }).to_parquet(db / "T3_30_300_spectral.parquet", index=False)


def make_trees(path: Path, kind: str = "polygons") -> Path:
    """Three trees around building B0: crowns (polygons with height) or bare points."""
    pts = [shapely.Point(X0 + 5, Y0 + 20), shapely.Point(X0 + 30, Y0 + 30), shapely.Point(X0 + 120, Y0 + 50)]
    if kind == "polygons":
        gdf = gpd.GeoDataFrame({"height": [6.0, 12.0, 20.0]}, geometry=[p.buffer(r) for p, r in zip(pts, (2, 3, 4))], crs=CRS)
        gdf.to_file(path, layer="trees")
    else:
        gpd.GeoDataFrame(geometry=pts, crs=CRS).to_crs("EPSG:4326").to_parquet(path, index=False)
    return path
