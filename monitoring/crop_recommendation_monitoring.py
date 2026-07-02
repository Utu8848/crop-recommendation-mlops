import os
import pickle
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv("/home/utsav/airflow/.env")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("crop_recommendation_monitoring")

CONFIG = {
    # Database
    "mariadb_host"          : "127.0.0.1",
    "mariadb_port"          : 3307,
    "mariadb_user"          : os.environ.get("MARIADB_USER",     ""),
    "mariadb_password"      : os.environ.get("MARIADB_PASSWORD", ""),
    "database_name"         : "crop_db",
    "obt_table"             : "obt_crop_features",
    "predictions_table"     : "predictions_log",

    # Redis
    "redis_host"            : "127.0.0.1",
    "redis_port"            : 6379,

    # MLflow
    "mlflow_tracking_uri"   : "http://127.0.0.1:5000",
    "experiment_name"       : "crop_recommendation_system",
    "registered_model_name" : "crop_recommendation_model",

    # Monitoring
    "min_prediction_rows"   : 50,
    "report_output_dir"     : "/home/utsav/airflow/monitoring_reports",
    "feature_cols"          : [
        "nitrogen", "phosphorus", "potassium",
        "temperature", "humidity", "ph", "rainfall"
    ],
}


def _get_db_url() -> str:
    u = CONFIG["mariadb_user"]
    p = CONFIG["mariadb_password"]
    h = CONFIG["mariadb_host"]
    n = CONFIG["mariadb_port"]
    d = CONFIG["database_name"]
    return f"mysql+pymysql://{u}:{p}@{h}:{n}/{d}"


def run_monitoring(window_hours: int = 24):
    """
    Run the full Evidently + KS monitoring suite against predictions_log.

    Parameters
    ----------
    window_hours : int
        How many hours back from now to treat as the current prediction window.
        Defaults to 24 (last hour's predictions). Pass 0 to use all available rows.
    """
    import redis as redis_lib
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sqlalchemy import create_engine
    from scipy      import stats as scipy_stats

    from evidently               import ColumnMapping
    from evidently.report        import Report
    from evidently.metric_preset import TargetDriftPreset, DataQualityPreset
    from evidently.metrics       import (
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnDriftMetric,
        ColumnDistributionMetric,
        ColumnSummaryMetric,
    )

    db_engine = create_engine(_get_db_url(), pool_pre_ping=True)
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])
    os.makedirs(CONFIG["report_output_dir"], exist_ok=True)

    feature_cols = CONFIG["feature_cols"]

    # Time window
    now          = datetime.utcnow()
    window_end   = now.strftime("%Y-%m-%d %H:%M:%S")
    window_start = (now - timedelta(hours=window_hours)).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Monitoring window: {window_start} → {window_end}")

    # Guard: enough total predictions?
    total_count = pd.read_sql(
        "SELECT COUNT(*) AS total FROM predictions_log "
        "WHERE response_status = 'success'",
        db_engine
    )["total"].iloc[0]
    logger.info(f"Total predictions in DB: {total_count}")

    if total_count < CONFIG["min_prediction_rows"]:
        logger.warning(
            f"Insufficient total predictions ({total_count} < "
            f"{CONFIG['min_prediction_rows']}) — monitoring skipped."
        )
        return

    # Load current window
    current_df = pd.read_sql(
        f"""
        SELECT {', '.join(feature_cols)},
               predicted_crop   AS target,
               confidence_score
        FROM {CONFIG['predictions_table']}
        WHERE response_status = 'success'
          AND prediction_timestamp BETWEEN '{window_start}' AND '{window_end}'
        ORDER BY prediction_timestamp DESC
        """,
        db_engine
    )
    window_used = f"last_{window_hours}h"

    if len(current_df) < CONFIG["min_prediction_rows"]:
        logger.warning(
            f"Only {len(current_df)} predictions in window — "
            f"falling back to all available predictions."
        )
        current_df = pd.read_sql(
            f"""
            SELECT {', '.join(feature_cols)},
                   predicted_crop   AS target,
                   confidence_score
            FROM {CONFIG['predictions_table']}
            WHERE response_status = 'success'
            ORDER BY prediction_timestamp DESC
            """,
            db_engine
        )
        window_used = "all_available"

    logger.info(f"Monitoring window: {window_used} — {len(current_df)} rows.")

    # Load OBT baseline
    baseline_df = pd.read_sql(
        f"SELECT {', '.join(feature_cols)}, label AS target "
        f"FROM {CONFIG['obt_table']} ORDER BY id",
        db_engine
    )
    logger.info(f"Baseline loaded: {len(baseline_df)} rows.")

    confidence_series    = current_df["confidence_score"].copy()
    current_df_evidently = current_df.drop(columns=["confidence_score"])

    for col in feature_cols:
        baseline_df[col]          = baseline_df[col].astype(float)
        current_df_evidently[col] = current_df_evidently[col].astype(float)

    baseline_df["target"]          = baseline_df["target"].astype(str)
    current_df_evidently["target"] = current_df_evidently["target"].astype(str)

    # KS statistics
    ks_results = {}
    for col in feature_cols:
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(
            baseline_df[col].values,
            current_df_evidently[col].values
        )
        ks_results[col] = {
            "statistic": round(float(ks_stat),   4),
            "pvalue"   : round(float(ks_pvalue), 4),
            "drifted"  : bool(ks_pvalue < 0.05)
        }

    n_ks_drifted = sum(1 for v in ks_results.values() if v["drifted"])
    logger.info(f"KS drift: {n_ks_drifted}/{len(feature_cols)} features drifted.")

    column_mapping = ColumnMapping(
        target               = "target",
        numerical_features   = feature_cols,
        categorical_features = []
    )

    ts = now.strftime("%Y%m%d_%H%M%S")

    # Evidently: data drift
    # Evidently: data drift
    drift_report = Report(metrics=[
        DatasetDriftMetric(),
        *[ColumnDriftMetric(column_name=col) for col in feature_cols],
    ])
    drift_report.run(
        reference_data = baseline_df,
        current_data   = current_df_evidently,
        column_mapping = column_mapping
    )
    drift_path = os.path.join(CONFIG["report_output_dir"], f"drift_report_{ts}.html")
    drift_report.save_html(drift_path)
    logger.info(f"Drift report saved: {drift_path}")

    # Evidently: target drift
    target_report = Report(metrics=[TargetDriftPreset()])
    target_report.run(
        reference_data = baseline_df,
        current_data   = current_df_evidently,
        column_mapping = column_mapping
    )
    target_path = os.path.join(CONFIG["report_output_dir"], f"target_drift_report_{ts}.html")
    target_report.save_html(target_path)
    logger.info(f"Target drift report saved: {target_path}")

    # Evidently: data quality
    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(
        reference_data = baseline_df,
        current_data   = current_df_evidently,
        column_mapping = column_mapping
    )
    quality_path = os.path.join(CONFIG["report_output_dir"], f"data_quality_report_{ts}.html")
    quality_report.save_html(quality_path)
    logger.info(f"Data quality report saved: {quality_path}")

    # Confidence drift — scaler from Redis, fallback to MLflow
    r_conf = redis_lib.Redis(
        host             = CONFIG["redis_host"],
        port             = CONFIG["redis_port"],
        decode_responses = False,
        socket_timeout   = 10
    )
    scaler_raw = r_conf.get("crop:scaler")
    if scaler_raw is None:
        logger.warning(
            "crop:scaler not found in Redis — loading from MLflow Production artifacts..."
        )
        model_artifacts_path = mlflow.artifacts.download_artifacts(
            f"models:/{CONFIG['registered_model_name']}@Production"
        )
        scaler_artifact_path = os.path.join(model_artifacts_path, "scaler.pkl")
        if os.path.exists(scaler_artifact_path):
            with open(scaler_artifact_path, "rb") as f:
                scaler_for_conf = pickle.load(f)
        else:
            raise RuntimeError(
                "crop:scaler not in Redis and no scaler.pkl in MLflow artifacts. "
                "Re-run the feature_engineering task."
            )
    else:
        scaler_for_conf = pickle.loads(scaler_raw)

    model_for_conf     = mlflow.sklearn.load_model(
        f"models:/{CONFIG['registered_model_name']}@Production"
    )
    X_baseline_raw     = baseline_df[feature_cols].values.astype(float)
    X_baseline_scaled  = scaler_for_conf.transform(X_baseline_raw)
    baseline_conf      = model_for_conf.predict_proba(X_baseline_scaled).max(axis=1)

    baseline_conf_df = pd.DataFrame({"confidence_score": baseline_conf.astype(float)})
    current_conf_df  = pd.DataFrame({"confidence_score": confidence_series.astype(float)})

    from evidently.metrics import ColumnDriftMetric as _CDM  # already imported above
    confidence_report = Report(metrics=[
        ColumnSummaryMetric(column_name="confidence_score"),
        ColumnDistributionMetric(column_name="confidence_score"),
        ColumnDriftMetric(column_name="confidence_score"),
    ])
    confidence_report.run(
        reference_data = baseline_conf_df,
        current_data   = current_conf_df,
        column_mapping = ColumnMapping(numerical_features=["confidence_score"])
    )
    confidence_path = os.path.join(
        CONFIG["report_output_dir"], f"confidence_report_{ts}.html"
    )
    confidence_report.save_html(confidence_path)
    logger.info(f"Confidence report saved: {confidence_path}")

    # Extract metrics
    def get_metric_result(report_dict, metric_class_name, column_name=None):
        for m in report_dict["metrics"]:
            if m["metric"] == metric_class_name:
                result = m["result"]
                if column_name is None or result.get("column_name") == column_name:
                    return result
        return {}

    drift_result   = drift_report.as_dict()
    dataset_result = get_metric_result(drift_result, "DatasetDriftMetric")
    drift_detected = dataset_result.get("dataset_drift",             False)
    n_drifted      = dataset_result.get("number_of_drifted_columns", 0)
    drift_share    = dataset_result.get("share_of_drifted_columns",  0.0)

    evidently_col_drift = {}
    for col in feature_cols:
        col_result = get_metric_result(drift_result, "ColumnDriftMetric", col)
        evidently_col_drift[col] = {
            "drift_score"    : round(col_result.get("drift_score",    0.0), 4),
            "drift_detected" : col_result.get("drift_detected",       False),
            "stattest_name"  : col_result.get("stattest_name",        "unknown"),
        }

    conf_result_dict    = confidence_report.as_dict()
    conf_drift_result   = get_metric_result(
        conf_result_dict, "ColumnDriftMetric", "confidence_score"
    )
    conf_drift_detected = conf_drift_result.get("drift_detected", False)
    conf_stats          = confidence_series.describe()

    logger.info(
        f"Monitoring complete — "
        f"drift_detected: {drift_detected}, "
        f"drifted_features: {n_drifted}/{len(feature_cols)}, "
        f"confidence_drift: {conf_drift_detected}, "
        f"window: {window_used}, "
        f"avg_confidence: {conf_stats['mean']:.4f}"
    )

    if drift_detected:
        logger.warning(
            f"DATA DRIFT DETECTED — "
            f"{n_drifted} of {len(feature_cols)} features drifted "
            f"(share: {drift_share:.2%}). "
            f"Consider triggering a retraining run. "
            f"Reports: {CONFIG['report_output_dir']}"
        )

    # Log to MLflow
    with mlflow.start_run(run_name=f"monitoring_{ts}"):
        mlflow.log_metric("drift_detected",            int(drift_detected))
        mlflow.log_metric("n_drifted_features",        n_drifted)
        mlflow.log_metric("drift_share",               round(drift_share,        4))
        mlflow.log_metric("n_predictions_monitored",   int(len(current_df)))
        mlflow.log_metric("avg_confidence",            round(conf_stats["mean"], 4))
        mlflow.log_metric("min_confidence",            round(conf_stats["min"],  4))
        mlflow.log_metric("std_confidence",            round(conf_stats["std"],  4))
        mlflow.log_metric("confidence_drift_detected", int(conf_drift_detected))

        for col, res in evidently_col_drift.items():
            mlflow.log_metric(f"evidently_drift_score_{col}", res["drift_score"])
            mlflow.log_metric(f"evidently_drifted_{col}",     int(res["drift_detected"]))

        for col, res in ks_results.items():
            mlflow.log_metric(f"ks_stat_{col}",   res["statistic"])
            mlflow.log_metric(f"ks_pvalue_{col}", res["pvalue"])
        mlflow.log_metric("ks_n_drifted_features", n_ks_drifted)

        mlflow.log_artifact(drift_path)
        mlflow.log_artifact(target_path)
        mlflow.log_artifact(quality_path)
        mlflow.log_artifact(confidence_path)

        mlflow.log_param("monitoring_window",  window_used)
        mlflow.log_param("baseline_source",    CONFIG["obt_table"])
        mlflow.log_param("baseline_rows",      len(baseline_df))
        mlflow.log_param("current_rows",       len(current_df))
        mlflow.log_param("window_start",       window_start)
        mlflow.log_param("window_end",         window_end)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop Recommendation — Monitoring")
    parser.add_argument(
        "--window-hours",
        type    = int,
        default = 24,
        help    = "Hours back from now to use as the prediction window (0 = all rows)"
    )
    args = parser.parse_args()
    run_monitoring(window_hours=args.window_hours)
