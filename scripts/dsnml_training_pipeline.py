from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

try:
    from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("catboost is required. Install dependencies via pip install -r requirements.txt") from exc

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("lightgbm is required. Install dependencies via pip install -r requirements.txt") from exc

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("xgboost is required. Install dependencies via pip install -r requirements.txt") from exc

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

    if "product_category" in df.columns:
        non_edible = ["household", "health and hygiene", "hard drinks", "others"]
        df["fat_content_adjusted"] = np.where(
            df["product_category"].isin(non_edible),
            "Non-Edible",
            df["fat_content"].fillna("Unknown"),
        )

    return df


def add_group_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create leakage-safe summary statistics from the training data only."""
    cat_price = train.groupby("product_category")["product_price"].mean()
    store_price = train.groupby("store_code")["product_price"].mean()
    store_format_sales = train.groupby("store_format")[TARGET].mean()
    product_sales_mean = train.groupby("product_code")[TARGET].mean()
    product_price_mean = train.groupby("product_code")["product_price"].mean()
    global_mean_price = train["product_price"].mean()
    global_target_mean = train[TARGET].mean()

    for df in [train, test]:
        df["price_to_cat_mean"] = df["product_price"] / df["product_category"].map(cat_price).fillna(global_mean_price)
        df["price_to_store_mean"] = df["product_price"] / df["store_code"].map(store_price).fillna(global_mean_price)
        df["store_format_sales_mean"] = df["store_format"].map(store_format_sales).fillna(global_target_mean)
        df["product_sales_mean"] = df["product_code"].map(product_sales_mean).fillna(global_target_mean)
        df["product_price_mean"] = df["product_code"].map(product_price_mean).fillna(global_mean_price)
        df["price_ratio_to_product_mean"] = df["product_price"] / df["product_code"].map(product_price_mean).fillna(global_mean_price)

        if "product_weight_kg" in df.columns:
            df["is_weight_missing"] = df["product_weight_kg"].isna().astype(int)
            df["product_weight_kg"] = df["product_weight_kg"].fillna(-1)

    return train, test


def get_model_inputs(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[int]]:
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
    return X, y, X_test, cat_features, cat_idx


def evaluate_catboost(X: pd.DataFrame, y: pd.Series, cat_idx: list[int], params: dict) -> tuple[float, float, np.ndarray]:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores: list[float] = []
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X):
        model = CatBoostRegressor(cat_features=cat_idx, random_seed=SEED, verbose=False, **params)
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


def evaluate_lightgbm(X: pd.DataFrame, y: pd.Series, categorical_cols: list[str]) -> tuple[float, float, np.ndarray]:
    X_eval = X.copy()
    for col in categorical_cols:
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype("category")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores: list[float] = []
    oof_preds = np.zeros(len(X_eval))

    for train_idx, val_idx in kf.split(X_eval):
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=SEED,
            verbosity=-1,
            objective="regression",
            metric="rmse",
        )
        model.fit(
            X_eval.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X_eval.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_pred = np.clip(model.predict(X_eval.iloc[val_idx]), 0, None)
        oof_preds[val_idx] = val_pred
        fold_scores.append(root_mean_squared_error(y.iloc[val_idx], val_pred))

    return float(np.mean(fold_scores)), float(np.std(fold_scores)), oof_preds


def evaluate_xgboost(X: pd.DataFrame, y: pd.Series, categorical_cols: list[str]) -> tuple[float, float, np.ndarray]:
    X_eval = X.copy()
    for col in categorical_cols:
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype("category")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores: list[float] = []
    oof_preds = np.zeros(len(X_eval))

    for train_idx, val_idx in kf.split(X_eval):
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            objective="reg:squarederror",
            tree_method="hist",
            enable_categorical=True,
            eval_metric="rmse",
        )
        model.fit(
            X_eval.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X_eval.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        val_pred = np.clip(model.predict(X_eval.iloc[val_idx]), 0, None)
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
    train, test = add_group_features(train, test)

    X, y, X_test, cat_features, cat_idx = get_model_inputs(train, test)

    experiment_log = pd.DataFrame(columns=["ID", "Change", "Model", "Target", "CV_RMSE", "Notes"])

    catboost_params = [
        {"iterations": 800, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_strength": 1},
        {"iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_strength": 1},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5, "random_strength": 2},
        {"iterations": 1000, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5, "random_strength": 1},
    ]

    best_result = {"model": "CatBoost", "rmse": float("inf"), "preds": None, "params": None}

    for i, params in enumerate(catboost_params, start=1):
        mean_rmse, std_rmse, oof_preds = evaluate_catboost(X, y, cat_idx, params)
        experiment_log.loc[len(experiment_log)] = [
            f"EXP-{i:03d}",
            "CatBoost tuning",
            "CatBoost",
            TARGET,
            f"{mean_rmse:.6f}",
            f"Std: {std_rmse:.6f}",
        ]
        print(f"CatBoost config {i}: RMSE {mean_rmse:.4f} +/- {std_rmse:.4f}")
        if mean_rmse < best_result["rmse"]:
            best_result = {"model": "CatBoost", "rmse": mean_rmse, "preds": oof_preds, "params": params}

    lgb_rmse, lgb_std, lgb_oof = evaluate_lightgbm(X, y, cat_features)
    experiment_log.loc[len(experiment_log)] = [
        "EXP-101",
        "LightGBM benchmark",
        "LightGBM",
        TARGET,
        f"{lgb_rmse:.6f}",
        f"Std: {lgb_std:.6f}",
    ]
    print(f"LightGBM benchmark: RMSE {lgb_rmse:.4f} +/- {lgb_std:.4f}")
    if lgb_rmse < best_result["rmse"]:
        best_result = {"model": "LightGBM", "rmse": lgb_rmse, "preds": lgb_oof, "params": None}

    xgb_rmse, xgb_std, xgb_oof = evaluate_xgboost(X, y, cat_features)
    experiment_log.loc[len(experiment_log)] = [
        "EXP-102",
        "XGBoost benchmark",
        "XGBoost",
        TARGET,
        f"{xgb_rmse:.6f}",
        f"Std: {xgb_std:.6f}",
    ]
    print(f"XGBoost benchmark: RMSE {xgb_rmse:.4f} +/- {xgb_std:.4f}")
    if xgb_rmse < best_result["rmse"]:
        best_result = {"model": "XGBoost", "rmse": xgb_rmse, "preds": xgb_oof, "params": None}

    print(f"Best model: {best_result['model']} with RMSE {best_result['rmse']:.4f}")

    train["oof_predictions"] = best_result["preds"]
    train["residual"] = train[TARGET] - train["oof_predictions"]
    pd.DataFrame({"id": train[ID_COL], "oof_predictions": train["oof_predictions"]}).to_csv(ROOT / "oof_predictions.csv", index=False)
    experiment_log.to_csv(ROOT / "experiment_log.csv", index=False)

    if best_result["model"] == "CatBoost":
        final_model = CatBoostRegressor(cat_features=cat_idx, random_seed=SEED, verbose=100, **best_result["params"])
        final_model.fit(X, y)
        final_predictions = np.clip(final_model.predict(X_test), 0, None)
    elif best_result["model"] == "LightGBM":
        X_model = X.copy()
        for col in cat_features:
            if col in X_model.columns:
                X_model[col] = X_model[col].astype("category")
        X_test_model = X_test.copy()
        for col in cat_features:
            if col in X_test_model.columns:
                X_test_model[col] = X_test_model[col].astype("category")
        final_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=SEED,
            verbosity=-1,
            objective="regression",
            metric="rmse",
        )
        final_model.fit(X_model, y)
        final_predictions = np.clip(final_model.predict(X_test_model), 0, None)
    else:
        X_model = X.copy()
        for col in cat_features:
            if col in X_model.columns:
                X_model[col] = X_model[col].astype("category")
        X_test_model = X_test.copy()
        for col in cat_features:
            if col in X_test_model.columns:
                X_test_model[col] = X_test_model[col].astype("category")
        final_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            objective="reg:squarederror",
            tree_method="hist",
            enable_categorical=True,
            eval_metric="rmse",
        )
        final_model.fit(X_model, y)
        final_predictions = np.clip(final_model.predict(X_test_model), 0, None)

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_predictions})
    submission.to_csv(ROOT / "final_catboost_submission.csv", index=False)
    print("Saved final_catboost_submission.csv")
    print("Saved experiment_log.csv")
    print("Saved oof_predictions.csv")


if __name__ == "__main__":
    main()
