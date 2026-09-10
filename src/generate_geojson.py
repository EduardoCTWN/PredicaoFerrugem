from pathlib import Path

import geopandas as gpd
import pandas as pd

from Helpers.municiples import load_municipalities


def generate_geojson(
    data: pd.DataFrame, output_path: str, base_date: pd.Timestamp
) -> str:
    """
    Join the consolidated predictions to the municipality polygons and
    write the GeoJSON used by the map.
    """
    # get the municipalities from the shapefile
    municipalities, _ = load_municipalities()

    # CD_MUN is the IBGE code: a stable merge key, unlike the row position.
    # NM_MUN comes in upper case, so title case reads better on the map.
    gdf = municipalities.rename(
        columns={"CD_MUN": "municipio_id", "NM_MUN": "municipio"}
    )
    gdf["municipio"] = gdf["municipio"].str.title()

    # merge with the predictions keeping every municipality by the left
    # parameter, so the map has no holes when a prediction is missing
    gdf = gdf.merge(data, on="municipio_id", how="left")

    # set the simulation date
    gdf["data_simulacao"] = base_date.strftime("%Y-%m-%d")

    # rename to the labels shown on the map tooltip. This has to come
    # before the columns are touched, otherwise creating a missing one
    # first would leave two columns with the same name after the rename.
    gdf = gdf.rename(
        columns={
            "dias_ate_chegada": "Dias ate a chegada da ferrugem",
            "data_chegada_prevista": "Data chegada prevista",
            "data_chegada_real": "Data chegada real",
        }
    )

    # prepare the date columns: ISO strings, and None where there is no
    # value so the driver writes null instead of the string "NaT"
    for col in ("Data chegada prevista", "Data chegada real"):
        if col not in gdf.columns:
            gdf[col] = None
            continue

        gdf[col] = pd.to_datetime(gdf[col]).dt.strftime("%Y-%m-%d")
        gdf[col] = gdf[col].astype(object).where(gdf[col].notna(), None)

    # the type Int64 supports NaN values so it does not convert to float
    days_col = "Dias ate a chegada da ferrugem"
    if days_col in gdf.columns:
        gdf[days_col] = gdf[days_col].astype("Int64")

    # drop vertices that sit within ~100m of the simplified line
    gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON", COORDINATE_PRECISION=5)

    return output_path


def generate_occurrences_geojson(
    occurrences: pd.DataFrame, output_path: str, season: str, base_date
) -> str:
    """
    Write the confirmed occurrences as points, one feature per report.
    """
    # keep only the season being mapped
    df = occurrences[occurrences["safra"] == season].copy()
    df["data"] = pd.to_datetime(df["data"])
    df = df[df["data"] <= base_date]
    # drop the data that do not have lat/long values - they cannot be placed in the map
    df = df.dropna(subset=["latitude", "longitude"])

    # create a geodataframe containing the information columns and turn the lat/long values into geometry points
    gdf = gpd.GeoDataFrame(
        df[["id", "data", "municipio", "estadio", "safra"]],
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # match the casing used on the polygon layer
    gdf["municipio"] = gdf["municipio"].str.title()

    # shown on the tooltip to make clear the point is a confirmed report,
    # not a prediction
    gdf["registro"] = "Ocorrência registrada pelo Consórcio Antiferrugem"

    gdf["data"] = pd.to_datetime(gdf["data"]).dt.strftime("%Y-%m-%d")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON", COORDINATE_PRECISION=5)

    return output_path


def generate_map_files(
    predictions: pd.DataFrame,
    occurrences: pd.DataFrame,
    output_dir,
    base_date: pd.Timestamp,
    season: str,
) -> tuple[str, str]:
    """
    Write the two layers the map consumes: municipality polygons with the
    predictions, and confirmed occurrences as points.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = base_date.strftime("%Y%m%d")

    polygons_path = generate_geojson(
        predictions, output_dir / f"municipios_{stamp}.geojson", base_date
    )
    points_path = generate_occurrences_geojson(
        occurrences, output_dir / f"ocorrencias_{stamp}.geojson", season, base_date
    )

    return str(polygons_path), str(points_path)
