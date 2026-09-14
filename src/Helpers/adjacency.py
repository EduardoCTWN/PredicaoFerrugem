from functools import lru_cache

import geopandas as gpd

from src.Helpers.municiples import load_municipalities

@lru_cache(maxsize=1)
def build_adjacency():
    """
    Return the neighbours of the municiples
    """
    mun, _ = load_municipalities()
    mun = mun.rename(columns={"CD_NUM": "municipio_id"})

    # touches cathces shared borders
    joined = gpd.sjoin(
        mun[["municipio_id", "geometry"]],
        mun[["municipio_id", "geometry"]],
        how="inner",
        predicate="touches",
    )

    return(
        joined.groupby("municipio_id_left")["municipio_id_right"].apply(list).to_dict
    )

def neighbours_alert(prediction, min_neighbours = 3):
    """
    When more than min_neighbours are in alert, the municiple gets flagged
    """

    adjacency = build_adjacency()
    df = prediction.sort_values(["data", "municipio_id"]).copy()

    for date, group in df.groupby("data"):
        in_alert = set(group.loc["trava_positiva"] == 1, "municipio_id")
        for idx, row in group[group["trava_positiva" == 0]].itterrows():
            neighbours = adjacency.get(row["municipio_id"], [])
            if sum(n in in_alert for n in neighbours) >= min_neighbours:
                df.at[idx, "trava_positiva"] = 1

    df["trava_positiva"] = (
        df.sort_values(["municipio_id", "data"])
        .groupby("municipio_id")["trava_positiva"]
        .cummax()
    )

    return df


