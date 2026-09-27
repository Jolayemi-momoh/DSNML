# DSNML training pipeline

This script trains a CatBoost regressor for the retail sales prediction task and writes a submission CSV.

Usage:
    python scripts/dsnml_training_pipeline.py

It expects the following files in the project root:
    - train.csv
    - test.csv

The script uses deterministic feature engineering and 5-fold cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor

SEED = 42
TARGET = "total_sales"
ID_COL = "id"


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


def main() -> None:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train = prepare_data(train)
    test = prepare_data(test)

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

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []
    all_oof = np.zeros(len(X))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_strength=1,
            cat_features=cat_idx,
            random_seed=SEED,
            verbose=False,
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[va_idx], y.iloc[va_idx]), early_stopping_rounds=50)
        val_pred = np.clip(model.predict(X.iloc[va_idx]), 0, None)
        all_oof[va_idx] = val_pred
        fold_score = root_mean_squared_error(y.iloc[va_idx], val_pred)
        fold_scores.append(fold_score)
        print(f"Fold {fold} RMSE: {fold_score:.4f}")

    print(f"Mean CV RMSE: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    final_model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_strength=1,
        cat_features=cat_idx,
        random_seed=SEED,
        verbose=100,
    )
    final_model.fit(X, y)

    final_predictions = np.clip(final_model.predict(X_test), 0, None)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_predictions})
    submission.to_csv("final_catboost_submission.csv", index=False)
    print("Saved final_catboost_submission.csv")


if __name__ == "__main__":
    main()
