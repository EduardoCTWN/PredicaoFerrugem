import geopandas as gpd
import pandas as pd

from Helpers.municiples import load_municipalities
from Helpers.season import season_of


def map_occurrences_to_municipalities(occurrences_path) -> pd.DataFrame:
    """
    Spatially join the consortium occurrences to the municipality
    polygons and return the first occurrence date per municipality
    and season.
    """
    df = pd.read_csv(occurrences_path, encoding="utf-8-sig")
    df["data"] = pd.to_datetime(df["data"])

    # drop the instances that do not have lat/long values
    df = df.dropna(subset=["latitude", "longitude"])

    # create a geo dataframe transforming the location points into geometry shapes
    occurrences = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # get the municipalities from the shapefile
    municipalities, _ = load_municipalities()
    municipalities = municipalities.rename(columns={"CD_MUN": "municipio_id"})

    # join the geodataframe from the real occurrences with the municipalities maintaining all the occurrences by the left parameter
    # the within parameter is used because the occurrences were reported in real localities
    joined = gpd.sjoin(
        occurrences,
        municipalities[["municipio_id", "geometry"]],
        how="left",
        predicate="within",
    )
    joined["safra"] = joined["data"].map(season_of)
    # the occurrences that were not joined are from outside the shapefile, so they don't get a municipio_id and they are reported here
    outside = joined["municipio_id"].isna().sum()
    if outside:
        print(f"{outside} occurrences that don't belong to shapefile")

    # The arrival date is the first confirmed occurrence of the season.
    arrival = (
        joined.dropna(subset=["municipio_id"])
        .groupby(["safra", "municipio_id"])["data"]
        .min()
        .rename("data_chegada_real")
        .reset_index()
    )

    # sjoin turns the string ids into float when it introduces NaN;
    # bring them back so the merge key matches.
    arrival["municipio_id"] = arrival["municipio_id"].astype(str)

    return arrival
