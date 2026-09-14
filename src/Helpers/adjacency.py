from functools import lru_cache

import geopandas as gpd

from Helpers.municiples import load_municipalities


@lru_cache(maxsize=1)
def build_adjacency() -> dict:
    """
    Map each municipality to the ones it shares a border with.

    Computed once: the topology does not change between runs, and the
    spatial join over 355 polygons is expensive.
    """
    mun, _ = load_municipalities()
    mun = mun.rename(columns={"CD_MUN": "municipio_id"})

    # "touches" catches shared borders without matching a polygon
    # against itself, which "intersects" would.
    joined = gpd.sjoin(
        mun[["municipio_id", "geometry"]],
        mun[["municipio_id", "geometry"]],
        how="inner",
        predicate="touches",
    )

    return (
        joined.groupby("municipio_id_left")["municipio_id_right"]
        .apply(list)
        .to_dict()
    )


def neighbours_alert(prediction, min_neighbours: int = 2):
    """
    Flag a municipality when enough of its neighbours are already
    flagged: rust spreads by spores, so a ring of active foci is itself
    evidence of risk.
    """
    if min_neighbours <= 0:
        return prediction

    adjacency = build_adjacency()
    df = prediction.sort_values(["data", "municipio_id"]).copy()

    for date, group in df.groupby("data"):
        # Snapshot taken before the loop: without it a single flagged
        # municipality could light up the whole state in one date.
        in_alert = set(group.loc[group["trava_positiva"] == 1, "municipio_id"])

        for idx, row in group[group["trava_positiva"] == 0].iterrows():
            neighbours = adjacency.get(row["municipio_id"], [])
            if sum(n in in_alert for n in neighbours) >= min_neighbours:
                df.at[idx, "trava_positiva"] = 1

    # The neighbour rule fires per date; the latch has to close again so
    # the alert survives to the following days.
    df["trava_positiva"] = (
        df.sort_values(["municipio_id", "data"])
        .groupby("municipio_id")["trava_positiva"]
        .cummax()
    )

    return df