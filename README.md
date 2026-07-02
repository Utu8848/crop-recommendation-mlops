# Crop Recommendation MLOps Pipeline

An end-to-end MLOps system that recommends the optimal crop to plant based on soil
and climate measurements (N, P, K, temperature, humidity, pH, rainfall). Built as a
production-style pipeline: automated ingestion, validation, feature engineering,
model training/registration, real-time serving, and drift monitoring.

## Architecture

- **Orchestration:** Apache Airflow (`dags/crop_recommendation_pipeline.py`)
- **Storage:** MariaDB ColumnStore (OLAP feature store) + InnoDB (OLTP prediction logs)
- **Feature/artifact cache:** Redis (train/test arrays, scaler, label encoder)
- **Experiment tracking & registry:** MLflow
- **Serving:** FastAPI (`api/app.py`)
- **Data validation:** Great Expectations
- **Model explainability:** SHAP
- **Drift monitoring:** Evidently + Kolmogorov–Smirnov tests (`monitoring/crop_recommendation_monitoring.py`)

## Pipeline stages

1. **Data Ingestion** — start/verify ColumnStore, create schema, validate CSV with Great Expectations, deduplicate via row hashing, bulk-load with `cpimport`.
2. **Data Preprocessing** — stratified train/test split, `StandardScaler` fit on train only (no leakage), label encoding, caching to Redis via Arrow/pickle.
3. **Model Training** — trains RandomForest and GradientBoosting, cross-validation, learning curves, confusion matrices, SHAP global importance, quality gate (min F1 = 0.90), promotes the best model to the MLflow `Production` alias.
4. **Serving** — FastAPI `/predict` endpoint loads the Production model, scaler, and label encoder at startup; logs every prediction to `predictions_log`.
5. **Monitoring** — scheduled job compares recent predictions against the training baseline using Evidently (data drift, target drift, data quality, confidence drift) and KS tests; logs results back to MLflow.

## Repository structure

```
crop_recommendation/
├── dags/
│   └── crop_recommendation_pipeline.py
├── api/
│   └── app.py
├── monitoring/
│   └── crop_recommendation_monitoring.py
├── data/
│   └── Crop_recommendation.csv        (gitignored, not tracked)
├── docs/
│   └── project report and appendices
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- Docker (for MariaDB ColumnStore container)
- Redis
- Apache Airflow
- MLflow tracking server

### Installation

```bash
git clone https://github.com/Utu8848/crop-recommendation-mlops.git
cd crop-recommendation-mlops
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# then fill in your own credentials
```

### Running the pipeline

Copy `dags/crop_recommendation_pipeline.py` into your Airflow `dags/` folder, then unpause `crop_recommendation_pipeline` in the Airflow UI (or CLI).

### Running the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Running monitoring manually

```bash
python monitoring/crop_recommendation_monitoring.py --window-hours 24
```

## Data

Source dataset: `Crop_recommendation.csv` (22 crop classes, 7 features: N, P, K, temperature, humidity, pH, rainfall).

## Model performance

Minimum acceptable test F1 (macro) for production promotion: **0.90**.
Best model is selected between RandomForest and GradientBoosting based on held-out test F1, with SHAP used to explain the winning model's feature importances.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

Utsav Rai
