import pandas as pd

def correct_model(prediction, arrival, season):
    arrival_season = arrival[arrival["safra"] == season]

    confirmed = dict(
        zip(
            arrival_season["municipio_id"],
            pd.to_datetime(arrival_season["data_chegada_real"]),
        )
    )

    df = prediction.copy()

    report_date = df["municipio_id"].map(confirmed)
    df["confirmado"] = (report_date.notna() & (df["data"] >= report_date)).astype(int)

    df["alerta_mapa"] = ((df["trava_positiva"] == 1) | (df["confirmado"] == 1)).astype(int)

    return df