from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

SEED = 42
TARGET = "total_sales"
ID_COL = "id"
ROOT = Path(__file__).resolve().parent.parent


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["product_category", "fat_content"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.lower().str.strip()

    if "product_weight_kg" in df.columns:
        df["product_weight_kg"] = df["product_weight_kg"].fillna(-1)
        df["weight_clean"] = df["product_weight_kg"].replace(-1, np.nan)
        if "product_price" in df.columns:
            df["price_per_kg"] = df["product_price"] / df["weight_clean"]

    if "store_size" in df.columns:
        df["store_size"] = df["store_size"].fillna("Unknown").astype(str)

    if {"store_location_tier", "store_format"}.issubset(df.columns):
        df["store_tier_format"] = df["store_location_tier"].astype(str) + "_" + df["store_format"].astype(str)

    non_edible = ["household", "health and hygiene", "hard drinks", "others"]
    if "product_category" in df.columns:
        df["fat_content_adjusted"] = np.where(
            df["product_category"].isin(non_edible),
            "Non-Edible",
            df["fat_content"].fillna("Unknown"),
        )

    return df


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[int]]:
    cat_features = [
        "product_code",
        "fat_content",
        "product_category",
        "store_code",
        "store_size",
        "store_location_tier",
        "store_format",
        "store_tier_format",
        "fat_content_adjusted",
    ]

    for col in cat_features:
        if col in train.columns:
            train[col] = train[col].fillna("Unknown").astype(str)
        if col in test.columns:
            test[col] = test[col].fillna("Unknown").astype(str)

    X = train.drop(columns=[ID_COL, TARGET])
    y = train[TARGET]
    X_test = test.drop(columns=[ID_COL])

    cat_idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]
    return X, y, X_test, cat_idx


def evaluate_catboost(X: pd.DataFrame, y: pd.Series, cat_idx: list[int], params: dict) -> tuple[float, float, np.ndarray]:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores: list[float] = []
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X):
        model = CatBoostRegressor(
            cat_features=cat_idx,
            random_seed=SEED,
            verbose=False,
            **params,
        )
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            early_stopping_rounds=50,
        )
        val_pred = np.clip(model.predict(X.iloc[val_idx]), 0, None)
        oof_preds[val_idx] = val_pred
        fold_scores.append(root_mean_squared_error(y.iloc[val_idx], val_pred))

    return float(np.mean(fold_scores)), float(np.std(fold_scores)), oof_preds


def main() -> None:
    train_path = ROOT / "train.csv"
    test_path = ROOT / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train.csv and test.csv in the repository root.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = prepare_data(train)
    test = prepare_data(test)

    X, y, X_test, cat_idx = prepare_features(train, test)

    tuning_grid = [
        {"iterations": 800, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_strength": 1},
        {"iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_strength": 1},
        {"iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5, "random_strength": 2},
        {"iterations": 1000, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5, "random_strength": 1},
    ]

    experiment_log = pd.DataFrame(columns=["ID", "Change", "Model", "Target", "CV_RMSE", "Notes"])

    best_rmse = float("inf")
    best_params: dict | None = None
    best_oof: np.ndarray | None = None

    for idx, params in enumerate(tuning_grid, start=1):
        mean_rmse, std_rmse, oof_preds = evaluate_catboost(X, y, cat_idx, params)
        print(f"Config {idx}: {params} -> CV RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")

        experiment_log.loc[len(experiment_log)] = [
            f"EXP-{idx:03d}",
            "CatBoost hyperparameter tuning",
            "CatBoost",
            TARGET,
            f"{mean_rmse:.6f}",
            f"Std: {std_rmse:.6f}",
        ]

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params
            best_oof = oof_preds

    if best_params is None:
        raise RuntimeError("No valid CatBoost configuration was evaluated.")

    print(f"Best configuration: {best_params} with CV RMSE {best_rmse:.4f}")

    train["oof_predictions"] = best_oof
    train["residual"] = train[TARGET] - train["oof_predictions"]

    pd.DataFrame({"id": train[ID_COL], "oof_predictions": train["oof_predictions"]}).to_csv(ROOT / "oof_predictions.csv", index=False)
    experiment_log.to_csv(ROOT / "experiment_log.csv", index=False)

    final_model = CatBoostRegressor(
        cat_features=cat_idx,
        random_seed=SEED,
        verbose=100,
        **best_params,
    )
    final_model.fit(X, y)

    final_preds = np.clip(final_model.predict(X_test), 0, None)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_preds})
    submission.to_csv(ROOT / "final_catboost_submission.csv", index=False)

    print("Saved: final_catboost_submission.csv")
    print("Saved: experiment_log.csv")
    print("Saved: oof_predictions.csv")


if __name__ == "__main__":
    main()
