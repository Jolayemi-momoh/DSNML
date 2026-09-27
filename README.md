# DSNML

This repository contains a retail sales forecasting workflow built around a CatBoost regression pipeline.

## Project overview

The goal is to predict `total_sales` using structured retail data, feature engineering, and cross-validation.

## Repository structure

- `notebooks/DSNML_GitHub_version.ipynb` — polished Colab notebook for exploration, modelling, and submission generation.
- `scripts/dsnml_training_pipeline.py` — local/Colab-ready training pipeline with cross-validation and experiment logging.
- `requirements.txt` — Python dependency list.
- `README.md` — project usage notes.

## Data files

Place the dataset files in the repository root before running the notebook or script:

- `train.csv`
- `test.csv`

The expected target column is:

- `total_sales`

## Quick start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training pipeline:

```bash
python scripts/dsnml_training_pipeline.py
```

This will produce:

- `final_catboost_submission.csv`
- `experiment_log.csv`
- `oof_predictions.csv`

## Notebook usage

Open the notebook in Google Colab or Jupyter, and run the cells in order.

## Notes

The project uses a deterministic feature engineering strategy and CatBoost with 5-fold cross-validation for robust retail forecasting.
