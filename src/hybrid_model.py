import os

import joblib
import numpy as np
import pandas as pd

from Helpers.mapping_real_occorrences import (
    map_occurrences_to_municipalities,
    season_of,
)
from features import season_start_of

MODELS_FOLD = "Models"
CLASSIFIER_NAME = "XGB_time_classifier_2022.pkl"
REGRESSOR_NAME = "xgb_harvest_regressor_2022.pkl"


def load_models():
    classifier_path = os.path.join(MODELS_FOLD, CLASSIFIER_NAME)
    regressor_path = os.path.join(MODELS_FOLD, REGRESSOR_NAME)

    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier not found: {classifier_path}")
    if not os.path.exists(regressor_path):
        raise FileNotFoundError(f"Regressor not found: {regressor_path}")

    def unwrap(path):
        # Newer runs store {"model": ..., "threshold": ...}; older ones
        # store the estimator directly.
        bundle = joblib.load(path)
        return bundle["model"] if isinstance(bundle, dict) else bundle

    return unwrap(classifier_path), unwrap(regressor_path)


def predict(
    df: pd.DataFrame, classifier_threshold, regressor_threshold
) -> pd.DataFrame:
    classifier, regressor = load_models()
    df = df.copy()

    for name, model in (("classificador", classifier), ("regressor", regressor)):
        missing = [c for c in model.feature_names_in_ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features for {name}: {missing}")

    # Regressor - when the rust arrives
    X_reg = df[list(regressor.feature_names_in_)]
    days = regressor.predict(X_reg).round()
    df["data_chegada_prevista"] = df["data"].map(season_start_of) + pd.to_timedelta(
        days, unit="D"
    )
    df["dias_ate_chegada"] = (df["data_chegada_prevista"] - df["data"]).dt.days

    # Classifier - probability for rust happening today
    X_class = df[list(classifier.feature_names_in_)]
    df["predito_prob"] = np.clip(classifier.predict(X_class), 0, 1)

    # lock - risk but the arrival is far
    initial_risk_mask = df["predito_prob"] >= classifier_threshold
    far_mask = df["dias_ate_chegada"] > regressor_threshold
    vetoed = far_mask & initial_risk_mask

    df.loc[vetoed, "predito_prob"] = 0.0
    if vetoed.any():
        print(f"{vetoed.sum()} alerts discarded by the lock")

    # final class
    df["classe_predita"] = (df["predito_prob"] >= classifier_threshold).astype(int)

    # if there is risk, the arrival is today
    validated_mask = df["classe_predita"] == 1
    df.loc[validated_mask, "data_chegada_prevista"] = df.loc[validated_mask, "data"]
    df.loc[validated_mask, "dias_ate_chegada"] = 0

    return df


def consolidate_by_municipalitie(prediction: pd.DataFrame) -> pd.DataFrame:
    return (
        prediction.sort_values("data")
        .drop_duplicates(subset="municipio_id", keep="last")
        .reset_index(drop=True)
    )


def apply_latch(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Once a municipality turns positive it stays positive for the rest of
    the season: rust does not un-arrive.
    """
    df = predictions.sort_values(["municipio_id", "data"]).copy()

    df["trava_positiva"] = df.groupby("municipio_id")["classe_predita"].cummax()

    return df


def run(
    features_path: str,
    occurrences_path: str,
    base_date,
    classifier_threshold,
    threshold_regressor,
) -> pd.DataFrame:
    df = pd.read_csv(features_path, sep=";")
    df["data"] = pd.to_datetime(df["data"])

    predictions = predict(df, classifier_threshold, threshold_regressor)
    predictions = apply_latch(df)

    consolidated = consolidate_by_municipalitie(predictions)

    # Ground truth from the consortium, joined only for evaluation.
    arrival = map_occurrences_to_municipalities(occurrences_path)
    consolidated = consolidated.merge(
        arrival[arrival["safra"] == season_of(base_date)],
        on="municipio_id",
        how="left",
    )

    return consolidated
