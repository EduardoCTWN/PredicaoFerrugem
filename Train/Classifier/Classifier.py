import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_recall_curve,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
)
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def load_data():
    """
    This function is used to load and prepare the data
    for training.
    """
    df = pd.read_csv("Train/features_SI.csv")
    df["data"] = pd.to_datetime(df["data"], format="%Y-%m-%d")
    df["data_ocorrencia"] = pd.to_datetime(df["data_ocorrencia"], format="%Y-%m-%d")

    # Define the harvest year.
    df["safra"] = np.where(
        df["data_ocorrencia"].dt.month >= 9,
        df["data_ocorrencia"].dt.year,
        df["data_ocorrencia"].dt.year - 1,
    )

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

    return df


#######################################################
#
# DIVIDING THE DATA BY HARVEST AND BALANCING IT.
#
#######################################################


def divide_by_harvest(df, test_harvest):
    """
    Divides the dataset for training and testing using
    harvest
    """

    # Takes the harvest that the function receives as a parameter and put in the test group
    # and the other harvests are put into the train.
    df_grouped = df.groupby("ocorrencia_id")
    df_grouped_test = [g for g in df_grouped if g[1]["safra"].iloc[0] == test_harvest]
    df_grouped_train = [g for g in df_grouped if g[1]["safra"].iloc[0] != test_harvest]

    if len(df_grouped_test) == 0:
        raise ValueError(f"The {test_harvest} harvest doesn't have data to test")

    if len(df_grouped_train) == 0:
        raise ValueError(f"No harvests left for training after removing {test_harvest}")

    df_train = pd.concat([group for _, group in df_grouped_train])
    df_test = pd.concat([group for _, group in df_grouped_test])

    return df_train, df_test


def balance(df_train):
    """
    Balance the training dataset between classes 0 and 1
    """

    target_1 = df_train[df_train["target"] == 1]
    # Randomly choose the same number of 0s as there are 1s
    target_0 = df_train[df_train["target"] == 0].sample(
        n=target_1.shape[0], random_state=52
    )

    return pd.concat([target_0, target_1])


#######################################################
#
# K-FOLD TRAINING
#
#######################################################


def find_best_threshold(y_true, y_pred, beta=1.0):
    """
    Return the threshold that maximizes the F-beta score on out-of-fold
    predictions. Beta above 1 favors recall, below 1 favors precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    # precision_recall_curve returns one more point than thresholds
    precision, recall = precision[:-1], recall[:-1]

    b2 = beta**2
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall + 1e-9)

    return float(thresholds[np.argmax(fbeta)])


# CHECK LATER THE ADDING OF THE GROUP PARAMETER ##############################################
def kfold_train(X, y, groups, report_file, model_type="rf", n_splits=5, beta=1.0):
    """
    Run stratified k-fold cross-validation with the chosen model and return the metrics and the final fitted model.
    """

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2s, maes, rmses = [], [], []
    oof_true, oof_pred = [], []

    def create_model():
        if model_type == "lgbm":
            return lgb.LGBMRegressor(
                objective="regression_l1",
                metric="mae",
                n_estimators=2000,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.5,
                random_state=42,
                verbosity=-1,
                n_jobs=8,
            )
        elif model_type == "xgb":
            ratio = 1.5
            return xgb.XGBRegressor(
                objective="binary:logistic",
                # scale_pos_weight=ratio,
                eval_metric="auc",
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.5,
                colsample_bylevel=0.6,
                reg_alpha=0.3,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=8,
                verbosity=0,
            )
        elif model_type == "rf":
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=8,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")

    # K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = create_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        r2s.append(r2_score(y_val, y_pred))
        maes.append(mean_absolute_error(y_val, y_pred))
        oof_true.append(y_val.to_numpy())
        oof_pred.append(y_pred)
        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    oof_true = np.concatenate(oof_true)
    oof_pred = np.concatenate(oof_pred)

    best_threshold = find_best_threshold(oof_true, oof_pred, beta)
    report_file.write(
        f"Out-of-fold threshold: {best_threshold:.4f} with beta = {beta}\n"
    )

    # Train the final model with all the data
    final_model = create_model()
    final_model.fit(X, y)

    feature_importance = calculate_feature_importance(final_model, X, report_file)

    return {
        "r2_mean": np.mean(r2s),
        "mae_mean": np.mean(maes),
        "rmse_mean": np.mean(rmses),
        "threshold": best_threshold,
        "model": final_model,
    }


#######################################################
#
# FEATURE IMPORTANCE
#
#######################################################


def calculate_feature_importance(model, X, report_file):
    """
    Calculate the importance of the features and print it
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importances_df = pd.DataFrame(
            {"feature": X.columns, "importance": importances}
        ).sort_values(by="importance", ascending=False)

        report_file.write("Importance of the features\n")
        report_file.write(feature_importances_df.head(15).to_string())
        report_file.write("\n")

        return feature_importances_df

    else:
        report_file.write(f"Model doesn't have feature importance\n")

        return None


#######################################################
#
# EVALUATE THE TEST HARVEST
#
#######################################################


def evaluate_harvest(model, df_test, threshold, temp_includes):
    # Drop the columns that won't be used in the test - to avoid leak
    cols_to_drop = ["ocorrencia_id", "data", "data_ocorrencia", "target", "safra"]
    if not temp_includes:
        cols_to_drop += ["dia_da_safra"]

    X_test_all = df_test.drop(columns=cols_to_drop)
    Y_test_all = df_test["target"]

    # It predicts once to save time - batch prediction
    Y_pred_all = model.predict(X_test_all)

    # Add the prediction to the dataframe for further operations
    df_results = df_test[["ocorrencia_id", "target"]].copy()
    df_results["pred"] = Y_pred_all

    errors, tp, fp, fn, tn = [], 0, 0, 0, 0

    # We group the results here just to calculate some stats
    for ocorrencia_id, group in df_results.groupby("ocorrencia_id"):
        Y_test = group["target"]
        y_pred = group["pred"]

        # Here we calculate the day error
        indexes_above_threshold = np.where(y_pred.values >= threshold)[0]
        if len(indexes_above_threshold) > 0:
            last_index = indexes_above_threshold[-1]
            errors.append(last_index)

        # confusion matrix - vectorized
        true_label = Y_test.astype(int)
        pred_label = (y_pred > threshold).astype(int)

        tp += ((true_label == 1) & (pred_label == 1)).sum()
        fp += ((true_label == 0) & (pred_label == 1)).sum()
        fn += ((true_label == 1) & (pred_label == 0)).sum()
        tn += ((true_label == 0) & (pred_label == 0)).sum()

    # Final stats
    r2 = r2_score(Y_test_all, Y_pred_all)
    mae = mean_absolute_error(Y_test_all, Y_pred_all)
    rmse = np.sqrt(mean_squared_error(Y_test_all, Y_pred_all))
    auc = roc_auc_score(Y_test_all, Y_pred_all)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        "day_error": np.mean(errors) if errors else np.nan,
        "r2_test": r2,
        "mae_test": mae,
        "rmse_test": rmse,
        "auc_test": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "precision": precision,
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
    model_name: str = "XGB_time_classifier",
    temp_includes: bool = True,
    output_path: str = None,
    beta: float = 1.0,
):
    df = load_data()

    # Every run gets its own directory, so models, report and results
    # stay together and past versions are never overwritten.
    run_id = (
        f"{execution_started_at:%Y%m%d_%H%M%S}_{model_type}"
        f"_beta{beta}_temp{int(temp_includes)}"
    )
    run_dir = os.path.join("Train/Classifier/Runs", run_id)

    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(run_dir, "results.csv")

    output_report_path = os.path.join(run_dir, "report.txt")

    if harvests is None:
        harvests = sorted(df["safra"].unique())

    results = []

    with open(output_report_path, "w", encoding="utf-8") as f:
        # Header with everything needed to reproduce this run.
        f.write(f"Run: {run_id}\n")
        f.write(f"Started at: {execution_started_at:%Y-%m-%d %H:%M:%S}\n")
        f.write(
            f"Model: {model_type} | beta: {beta} | temp_includes: {temp_includes}\n"
        )

        for test_harvest in harvests:
            f.write(f"\n------------ Testing {test_harvest} harvest ------------\n")
            print(f"Processing {test_harvest}...")

            try:
                df_train, df_test = divide_by_harvest(df, test_harvest)
            except ValueError as e:
                f.write(f"Skipping {test_harvest}: {e}\n")
                continue

            # df_train = balance(df_train)

            groups = df_train["ocorrencia_id"]

            drop_cols = ["ocorrencia_id", "data", "data_ocorrencia", "target", "safra"]
            if not temp_includes:
                drop_cols += ["dia_da_safra"]

            X = df_train.drop(columns=drop_cols)
            y = df_train["target"]

            metrics_kfold = kfold_train(X, y, groups, f, model_type, beta=beta)

            file_name_model = f"{model_name}_{test_harvest}.pkl"
            full_path_models = os.path.join(models_dir, file_name_model)

            # The threshold travels with the model, so inference never has
            # to guess which cut this model was calibrated for.
            joblib.dump(
                {
                    "model": metrics_kfold["model"],
                    "threshold": metrics_kfold["threshold"],
                    "beta": beta,
                    "temp_includes": temp_includes,
                },
                full_path_models,
            )
            f.write(f"Model {full_path_models} saved\n\n")

            f.write("Average stats for Kfold training:\n")
            f.write(f"Average R2: {metrics_kfold['r2_mean']:.4f}\n")
            f.write(f"Average MAE: {metrics_kfold['mae_mean']:.4f}\n")
            f.write(f"Average RMSE: {metrics_kfold['rmse_mean']:.4f}\n")

            metrics_test = evaluate_harvest(
                metrics_kfold["model"],
                df_test,
                metrics_kfold["threshold"],
                temp_includes,
            )

            f.write(f"\nFinal evaluation of {test_harvest} harvest\n")
            for k, v in metrics_test.items():
                linha = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                f.write(linha + "\n")

            results.append(
                {
                    "year": test_harvest,
                    "model_name": model_name,
                    "model_path": full_path_models,
                    "average_day_error": metrics_test["day_error"],
                    "average_R2": metrics_kfold["r2_mean"],
                    "average_MAE": metrics_kfold["mae_mean"],
                    "average_RMSE": metrics_kfold["rmse_mean"],
                    "average_AUC": metrics_test["auc_test"],
                    "TP": metrics_test["tp"],
                    "FP": metrics_test["fp"],
                    "FN": metrics_test["fn"],
                    "TN": metrics_test["tn"],
                    "average_Recall": metrics_test["recall"],
                    "average_Precision": metrics_test["precision"],
                    "threshold": metrics_kfold["threshold"],
                    "beta": beta,
                    "temp_includes": temp_includes,
                    "run_id": run_id,
                }
            )

        df_results = pd.DataFrame(results)

        df_results.to_csv(output_path, index=False)

        write_summary(df_results, f)

    append_to_index(run_id, df_results, model_type, beta, temp_includes)
    print(f"\nRun saved to {run_dir}")

    return df_results


def append_to_index(run_id, df_results, model_type, beta, temp_includes):
    """
    Append one summary row per run to a central index, so runs can be
    compared without opening each report.
    """
    if df_results.empty:
        return

    index_path = "Train/Classifier/runs_index.csv"

    row = {
        "run_id": run_id,
        "model_type": model_type,
        "beta": beta,
        "temp_includes": temp_includes,
        "mean_threshold": df_results["threshold"].mean(),
        "mean_AUC": df_results["average_AUC"].mean(),
        "mean_precision": df_results["average_Precision"].mean(),
        "mean_recall": df_results["average_Recall"].mean(),
        "mean_day_error": df_results["average_day_error"].mean(),
    }

    pd.DataFrame([row]).to_csv(
        index_path,
        mode="a",
        header=not os.path.exists(index_path),
        index=False,
    )


def write_summary(df_results, output_file):
    output_file.write("\n-------- Final average results --------\n")
    output_file.write(
        "Average R2: {:.3f}, Average MAE: {:.2f}, Average RMSE: {:.2f}, Average AUC: {:.3f},  "
        "Average precision: {:.2f}, Average recall: {:.2f}, average day error {:.2f}\n".format(
            df_results["average_R2"].mean(),
            df_results["average_MAE"].mean(),
            df_results["average_RMSE"].mean(),
            df_results["average_AUC"].mean(),
            df_results["average_Precision"].mean(),
            df_results["average_Recall"].mean(),
            df_results["average_day_error"].mean(),
        )
    )


if __name__ == "__main__":
    run(datetime.now(), beta=2.0)
