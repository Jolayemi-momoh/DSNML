# DSNML

This repository contains a retail sales forecasting workflow built around a CatBoost regressor pipeline.

## Project overview

The goal is to predict `total_sales` from retail data using structured tabular features, targeted feature engineering, and cross-validation.

## Repository structure

- `notebooks/DSNML_GitHub_version.ipynb` – interactive Colab notebook for exploration, feature engineering, modelling, and submission generation.
- `scripts/dsnml_training_pipeline.py` – standalone Python training script that can run in a local environment or Colab.
- `README.md` – repo instructions and usage guide.

## Data requirements

Place the following files in the repository root or update the paths in the script:

- `train.csv`
- `test.csv`

Expected columns include:

- `id`
- `product_code`
- `product_category`
- `fat_content`
- `product_weight_kg`
- `product_price`
- `store_code`
- `store_size`
- `store_location_tier`
- `store_format`
- `total_sales`

## Notebook usage

1. Open the notebook in Google Colab.
2. Upload or mount your dataset files.
3. Run the cells in order.
4. The notebook generates a Kaggle-style submission file named `final_catboost_submission.csv`.

## Script usage

From the repo root:

```bash
python scripts/dsnml_training_pipeline.py
```

The script will:

- load `train.csv` and `test.csv`
- perform basic cleaning and deterministic feature engineering
- train a CatBoost regressor using 5-fold cross-validation
- save `final_catboost_submission.csv`

## Environment

Recommended packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost
```

## Notes

This project is intended as a practical retail sales modeling workflow and is suitable for experimentation, CV benchmarking, and submission generation.
