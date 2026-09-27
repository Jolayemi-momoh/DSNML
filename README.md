# DSNML

This repository contains a retail sales forecasting workflow built around a benchmarked CatBoost, LightGBM, and XGBoost pipeline.

## Project overview

The goal is to predict `total_sales` from retail data using structured tabular features, targeted feature engineering, and cross-validation.

## Repository structure

- `notebooks/DSNML_GitHub_version.ipynb` — interactive Colab notebook for exploration, feature engineering, and model benchmarking.
- `scripts/dsnml_training_pipeline.py` — training script for benchmarking and exporting a Kaggle-style submission.
- `requirements.txt` — exact dependencies used by the project.
- `README.md` — repository guide.

## Data files

Place the following files in the repository root before running the notebook or script:

- `train.csv`
- `test.csv`

The target variable is:

- `total_sales`

## Quick start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the benchmark pipeline:

```bash
python scripts/dsnml_training_pipeline.py
```

This will generate the following output files in the repository root:

- `final_catboost_submission.csv`
- `oof_predictions.csv`
- `experiment_log.csv`

## Notebook usage

Open the notebook in Google Colab or Jupyter and run the cells in order. It covers:

- dataset preview and diagnostics
- engineered features
- CV evaluation across multiple models
- residual analysis
- submission generation

## Notes

This project is intended for experimentation and robust model comparison. It is suitable for use in local Python environments, Colab, and Kaggle competition-style submission workflows.
