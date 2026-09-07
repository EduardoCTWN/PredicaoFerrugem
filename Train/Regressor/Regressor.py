import pandas as pd
import numpy as np
from datetime import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
)
from xgboost import XGBRegressor

import os, joblib


def load_data():
    """
    This function is used to load and prepare the data
    for training.
    """
    df = pd.read_csv("Train/features_SI.csv")
    df["data"] = pd.to_datetime(df["data"], format="%Y-%m-%d")
    df["data_ocorrencia"] = pd.to_datetime(df["data_ocorrencia"], format="%Y-%m-%d")

    # Define the harvest year
    df["safra"] = np.where(
        df["data_ocorrencia"].dt.month >= 9,
        df["data_ocorrencia"].dt.year,
        df["data_ocorrencia"].dt.year - 1,
    )
    # Determine the begin of the harvest - {year}-09-01
    safra_start = pd.to_datetime(df["safra"].astype(str) + "-09-01")

    # target calculation and aditional features
    df["target"] = (df["data_ocorrencia"] - safra_start).dt.days

    # Any column that might have something to do with the planting
    cols_drop = [
        c
        for c in df.columns
        if any(
            p in c.lower()
            for p in (
                "dia_plantio",
                "dias_desde_plantio",
                "data_plantio",
                "plantio",
                "estadio",
                "estágio",
                "estagio",
                "fenolog",
            )
        )
        and c not in ("data_ocorrencia", "dia_da_safra")
    ]

    df = df.drop(columns=cols_drop)

    # CHECK IF IT IS NECESSARY
    df = df.sort_values("data").groupby("ocorrencia_id").tail(1).reset_index(drop=True)

    return df


#######################################################
#
# TRAIN AND TEST
#
#######################################################


def train_and_evaluate(
    df, test_harvest, report_file, model_type="rf", temp_includes=True
):
    """
    Train and evalueate the model using one harvest as a test
    """

    # Split the dataframe between train and test, using the harvest that we received as a
    # parameter to test
    # The columns "safra" is no longer necessary, so we drop it
    df_train = (
        df[df["safra"] != test_harvest].reset_index(drop=True).drop(columns=["safra"])
    )
    df_test = (
        df[df["safra"] == test_harvest].reset_index(drop=True).drop(columns=["safra"])
    )

    # Choosing wich features go, to avoid leak
    drop_cols = ["ocorrencia_id", "data", "data_ocorrencia", "target"]
    if not temp_includes:
        drop_cols.append("dia_da_safra")

    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    y_train = df_train["target"]

    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    y_test = df_test["target"]

    # Model choosing
    if model_type.lower() == "xgb":
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "xgb_harvest_regressor"
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model_name = "rf_harvest_regressor"

    report_file.write(
        f"\n------------ Testing {test_harvest} harvest with {model_name} ------------\n\n"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, test_harvest, model_name, report_file)

    return model, metrics


#######################################################
#
# Metrics
#
#######################################################


def compute_metrics(y_true, y_pred, test_harvest, model_name, report_file):
    """
    Calculate all the metrics
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    tolerance = 3
    diffs = y_pred - y_true
    TP = np.sum(np.abs(diffs) <= tolerance)
    FP = np.sum(diffs < -tolerance)
    FN = np.sum(diffs > tolerance)
    VN = len(y_true) - (TP + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    report_file.write(
        f"Harvest {test_harvest} → "
        f"R²: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, "
        f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}\n\n"
    )

    return {
        "year": test_harvest,
        "model_name": model_name,
        "average_R2": r2,
        "average_MAE": mae,
        "average_RMSE": rmse,
        "MedAE": medae,
        "VP": TP,
        "FP": FP,
        "FN": FN,
        "VN": VN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


#######################################################
#
# MAIN PIPELINE
#
#######################################################


def run(
    execution_started_at: datetime,
    harvests: list = None,
    model_type: str = "xgb",
    temp_includes: bool = True,
    model_name: str = None,
    output_path: str = None,
):
    df = load_data()

    models_dir = "Train/Regressor/Trained_regressors"
    os.makedirs(models_dir, exist_ok=True)

    output_report_dir = "Train/Regressor/Reports"
    os.makedirs(output_report_dir, exist_ok=True)

    results_dir = "Train/Regressor/Results"
    os.makedirs(results_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(
            results_dir, f"results_regressor_{execution_started_at:%Y%m%d_%H%M%S}.csv"
        )

    output_report_path = os.path.join(
        output_report_dir,
        f"regressor_report_{execution_started_at:%Y-%m-%d_%H%M%S}.txt",
    )

    unique_harvests = sorted(df["safra"].unique()) if harvests is None else harvests

    results = []
    with open(output_report_path, "w", encoding="utf-8") as f:
        for harvest in unique_harvests:
            print(f"processing {harvest}...")
            model, res = train_and_evaluate(df, harvest, f, model_type, temp_includes)
            res["exec_timestamp"] = execution_started_at.strftime("%Y-%m-%d %H:%M:%S")

            model_name = f"{model_type}_harvest_regressor"
            file_name_model = f"{model_name}_{harvest}.pkl"
            full_path_models = os.path.join(models_dir, file_name_model)
            res["model_path"] = full_path_models

            joblib.dump(model, full_path_models)
            f.write(f"Model {full_path_models} saved\n\n")

            results.append(res)

        df_results = pd.DataFrame(results)

        # Create the columns "day error" with MAE but could be with another metric
        df_results["day_error"] = df_results["average_MAE"]

        sorted_columns = [
            "year",
            "model_name",
            "model_path",
            "day_error",
            "average_R2",
            "average_MAE",
            "average_RMSE",
            "VP",
            "FP",
            "FN",
            "VN",
            "Recall",
            "Precision",
            "F1",
            "MedAE",
        ]
        results_df = df_results[sorted_columns]

        results_df.to_csv(output_path, index=False)

        write_summary(results_df, f)

    return results_df


def write_summary(df_results, output_file):
    output_file.write("\n-------- Final average results --------\n")
    output_file.write(
        "Average R2: {:.3f}, Average MAE: {:.2f}, Average RMSE: {:.2f}, Average MedAE {:.2f}  "
        "Average precision: {:.2f}, Average recall: {:.2f}, average day error {:.2f}\n".format(
            df_results["average_R2"].mean(),
            df_results["average_MAE"].mean(),
            df_results["average_RMSE"].mean(),
            df_results["MedAE"].mean(),
            df_results["Precision"].mean(),
            df_results["Recall"].mean(),
            df_results["day_error"].mean(),
        )
    )


if __name__ == "__main__":
    run(datetime.now(), temp_includes=False)
