from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors

EPSG_GEOGRAPHIC = 4326
EPSG_PROJECTED = 31982

BASE_DIR = Path(__file__).resolve().parent
SHAPEFILE_PATH = BASE_DIR / "Soil_limits" / "limite_municipios_soja.shp"


@lru_cache(maxsize=1)
def load_municipalities():
    """
    Read the shapefile and calculate the centroids
    """
    path = SHAPEFILE_PATH

    municipalities = gpd.read_file(path)
    assert municipalities.crs is not None
    if municipalities.crs.to_epsg() != EPSG_GEOGRAPHIC:
        municipalities = municipalities.to_crs(epsg=EPSG_GEOGRAPHIC)

    centroids = municipalities.to_crs(epsg=EPSG_PROJECTED).geometry.centroid.to_crs(
        epsg=EPSG_GEOGRAPHIC
    )
    coords = np.array([(p.y, p.x) for p in centroids])  # type: ignore

    print(f"shapefile: {len(municipalities)} municipalities")
    return municipalities, coords


def load_municipalities_as_points(
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame that contains all the municipalities of parana
    """

    municipalities, coords = load_municipalities()

    gdf = municipalities.rename(
        columns={"CD_MUN": "municipio_id", "NM_MUN": "municipio"}
    )

    gdf["poligono"] = gdf.geometry

    gdf[lat_col] = coords[:, 0]
    gdf[lon_col] = coords[:, 1]

    gdf = gdf.set_geometry(
        gpd.points_from_xy(gdf[lon_col], gdf[lat_col]),
        crs=f"EPSG:{EPSG_GEOGRAPHIC}",
    )

    return gdf
