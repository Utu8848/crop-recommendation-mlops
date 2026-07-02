import os
import logging
from datetime  import timedelta

from pendulum           import datetime as pdatetime
from dotenv             import load_dotenv
from airflow            import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

load_dotenv("/home/utsav/airflow/.env")

logger = logging.getLogger(__name__)

CONFIG = {
    # Database
    "mariadb_host"          : "127.0.0.1",
    "mariadb_port"          : 3307,
    "mariadb_user"          : os.environ.get("MARIADB_USER",     ""),
    "mariadb_password"      : os.environ.get("MARIADB_PASSWORD", ""),
    "database_name"         : "crop_db",
    "obt_table"             : "obt_crop_features",
    "predictions_table"     : "predictions_log",
    "columnstore_container" : "mymcs",
    "staging_dir"           : "/tmp/columnstore_staging",

    # Data
    "csv_path"              : "/home/utsav/airflow/dags/Crop_recommendation.csv",
    "data_source"           : "Crop_recommendation.csv",

    # Redis
    "redis_host"            : "127.0.0.1",
    "redis_port"            : 6379,
    "redis_ttl_training"    : 86400,
    "redis_ttl_artifacts"   : 86400 * 7,

    # MLflow
    "mlflow_tracking_uri"   : "http://127.0.0.1:5000",
    "experiment_name"       : "crop_recommendation_system",
    "registered_model_name" : "crop_recommendation_model",

    # Training
    "test_size"             : 0.2,
    "random_state"          : 42,
    "cv_folds"              : 5,

    # Monitoring
    "min_prediction_rows"   : 50,
    "report_output_dir"     : "/home/utsav/airflow/monitoring_reports",
    "feature_cols"          : [
        "nitrogen", "phosphorus", "potassium",
        "temperature", "humidity", "ph", "rainfall"
    ],
    "known_labels"          : [
        "apple", "banana", "blackgram", "chickpea", "coconut", "coffee",
        "cotton", "grapes", "jute", "kidneybeans", "lentil", "maize",
        "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya",
        "pigeonpeas", "pomegranate", "rice", "watermelon"
    ],
    "feature_ranges"        : {
        "nitrogen"    : (0,   140),
        "phosphorus"  : (5,   145),
        "potassium"   : (5,   205),
        "temperature" : (0,    50),
        "humidity"    : (14,  100),
        "ph"          : (3.5,10.0),
        "rainfall"    : (20, 3000),
    },
    
    "artifact_dir" : "/home/utsav/airflow/training_artifacts",
}

_missing_env = [k for k in ("MARIADB_USER", "MARIADB_PASSWORD") if not os.environ.get(k)]
if _missing_env:
    raise EnvironmentError(
        f"Required environment variables not set: {_missing_env}. "
        f"Check that /home/utsav/airflow/.env exists and is readable."
    )

def _on_failure_callback(context):
    """
    Called by Airflow when any task fails.
    Logs structured failure info — extend this to send Slack/email/PagerDuty.
    """
    task_id   = context["task_instance"].task_id
    dag_id    = context["task_instance"].dag_id
    exec_date = context["execution_date"]
    exception = context.get("exception")
    logging.getLogger(__name__).error(
        f"TASK FAILED | DAG: {dag_id} | Task: {task_id} | "
        f"execution_date: {exec_date} | Exception: {exception}"
    )

default_args = {
    "owner"               : "utsav",
    "depends_on_past"     : False,
    "retries"             : 1,
    "retry_delay"         : timedelta(minutes=2),
    "email_on_failure"    : False,
    "email_on_retry"      : False,
    "execution_timeout"   : timedelta(hours=2),
    "on_failure_callback" : _on_failure_callback,
}

def _get_db_urls():
    """Return (base_url, db_url) built from CONFIG at call time."""
    user = CONFIG["mariadb_user"]
    pwd  = CONFIG["mariadb_password"]
    host = CONFIG["mariadb_host"]
    port = CONFIG["mariadb_port"]
    db   = CONFIG["database_name"]
    base = f"mysql+pymysql://{user}:{pwd}@{host}:{port}"
    full = f"{base}/{db}"
    return base, full


# TASK GROUP 1 — DATA INGESTION

def check_and_start_columnstore():
    import subprocess
    import time

    container = CONFIG["columnstore_container"]

    logger.info("Checking ColumnStore status...")
    status_result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", "dbrmctl status"],
        capture_output=True, text=True, timeout=30
    )

    if "OK" in status_result.stdout:
        logger.info("ColumnStore already running.")
    else:
        logger.warning("ColumnStore not running — starting now...")
        subprocess.run(
            [
                "docker", "exec", container, "bash", "-c",
                """
                /usr/bin/mcs-loadbrm.py no > /dev/null 2>&1
                /usr/bin/workernode DBRM_Worker1 > /var/log/mariadb/columnstore/workernode.log 2>&1 &
                sleep 3
                /usr/bin/controllernode fg > /var/log/mariadb/columnstore/controllernode.log 2>&1 &
                sleep 3
                dbrmctl readwrite
                /usr/bin/PrimProc > /var/log/mariadb/columnstore/primproc.log 2>&1 &
                sleep 3
                /usr/bin/WriteEngineServer > /var/log/mariadb/columnstore/writeengineserver.log 2>&1 &
                sleep 3
                /usr/bin/DMLProc > /var/log/mariadb/columnstore/dmlproc.log 2>&1 &
                sleep 2
                /usr/bin/DDLProc > /var/log/mariadb/columnstore/ddlproc.log 2>&1 &
                sleep 5
                echo "ColumnStore Ready!"
                """
            ],
            check=True,
            timeout=120
        )
        time.sleep(5)
        logger.info("ColumnStore started.")

    # System catalog integrity check
    cat_result = subprocess.run(
        [
            "docker", "exec", container, "bash", "-c",
            "mysql -u root -e 'SELECT COUNT(*) FROM calpontsys.systable;' 2>/dev/null"
        ],
        capture_output=True, text=True, timeout=30
    )

    if "0" in cat_result.stdout or cat_result.returncode != 0:
        logger.warning("System catalog empty — rebuilding via dbbuilder 7...")
        db_result = subprocess.run(
            ["docker", "exec", container, "bash", "-c", "/usr/bin/dbbuilder 7"],
            capture_output=True, text=True, timeout=60
        )
        if db_result.returncode != 0:
            logger.warning(
                f"dbbuilder exited {db_result.returncode} — continuing. "
                f"stderr: {db_result.stderr}"
            )
        else:
            logger.info("System catalog rebuilt.")
        time.sleep(5)
    else:
        logger.info("System catalog OK.")

    final = subprocess.run(
        ["docker", "exec", container, "bash", "-c", "dbrmctl status"],
        capture_output=True, text=True, timeout=30
    )
    if "OK" not in final.stdout:
        raise RuntimeError(
            "ColumnStore did not start correctly. "
            "Check container logs: docker logs mymcs"
        )
    logger.info("ColumnStore confirmed running.")


def create_schema():
    from sqlalchemy import create_engine, text
    import time

    logger.info("Waiting 5s for ColumnStore DDLProc stabilization...")
    time.sleep(5)

    base_url, db_url = _get_db_urls()
    base_engine = create_engine(base_url, pool_pre_ping=True)
    db_engine   = create_engine(db_url,   pool_pre_ping=True)

    # Create database
    with base_engine.begin() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS {CONFIG['database_name']}"
        ))
    logger.info(f"Database '{CONFIG['database_name']}' ready.")

    # Create obt_crop_features (ColumnStore — OLAP bulk reads)
    with db_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS obt_crop_features (
                id                   BIGINT          NOT NULL,
                nitrogen             DECIMAL(10, 2)  NOT NULL,
                phosphorus           DECIMAL(10, 2)  NOT NULL,
                potassium            DECIMAL(10, 2)  NOT NULL,
                temperature          DECIMAL(10, 4)  NOT NULL,
                humidity             DECIMAL(10, 4)  NOT NULL,
                ph                   DECIMAL(10, 4)  NOT NULL,
                rainfall             DECIMAL(10, 4)  NOT NULL,
                label                VARCHAR(32)     NOT NULL,
                data_source          VARCHAR(64)     NOT NULL,
                ingestion_timestamp  DATETIME        NOT NULL,
                row_hash             CHAR(64)        NOT NULL
            ) ENGINE=ColumnStore
        """))
    logger.info("obt_crop_features ready (ENGINE=ColumnStore).")

    # Create predictions_log (InnoDB — OLTP single-row inserts from FastAPI)
    with db_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions_log (
                id                    BIGINT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
                request_id            VARCHAR(64)     NOT NULL,
                nitrogen              DECIMAL(10, 2)  NOT NULL,
                phosphorus            DECIMAL(10, 2)  NOT NULL,
                potassium             DECIMAL(10, 2)  NOT NULL,
                temperature           DECIMAL(10, 4)  NOT NULL,
                humidity              DECIMAL(10, 4)  NOT NULL,
                ph                    DECIMAL(10, 4)  NOT NULL,
                rainfall              DECIMAL(10, 4)  NOT NULL,
                predicted_crop        VARCHAR(32)     NOT NULL,
                confidence_score      DECIMAL(6, 4)   NOT NULL,
                model_version         VARCHAR(64)     NOT NULL,
                prediction_timestamp  DATETIME        NOT NULL,
                response_status       VARCHAR(16)     NOT NULL
            ) ENGINE=InnoDB
        """))
    logger.info("predictions_log ready (ENGINE=InnoDB).")


def validate_and_ingest():
    import os
    import hashlib
    import subprocess
    import time
    import pandas as pd
    import great_expectations as gx
    from datetime   import datetime
    from sqlalchemy import create_engine, text

    _, db_url = _get_db_urls()
    db_engine = create_engine(db_url, pool_pre_ping=True)
    container = CONFIG["columnstore_container"]

    # DMLProc health check
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", "pgrep -x DMLProc"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        logger.warning("DMLProc not running — restarting...")
        subprocess.run(
            [
                "docker", "exec", container, "bash", "-c",
                """
                killall -9 DMLProc 2>/dev/null
                sleep 2
                /usr/bin/DMLProc > /var/log/mariadb/columnstore/dmlproc.log 2>&1 &
                sleep 5
                """
            ],
            check=True, timeout=30
        )
        time.sleep(5)

    # Load and normalise CSV
    logger.info(f"Loading CSV: {CONFIG['csv_path']}")
    df = pd.read_csv(CONFIG["csv_path"])
    df = df.rename(columns={
        "N": "nitrogen", "P": "phosphorus", "K": "potassium",
        "temperature": "temperature", "humidity": "humidity",
        "ph": "ph", "rainfall": "rainfall", "label": "label"
    })
    logger.info(f"CSV loaded: {len(df)} rows.")

    # Great Expectations validation gate
    context    = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("crop_datasource")
    data_asset = datasource.add_dataframe_asset("raw_crop_data")
    batch_def  = data_asset.add_batch_definition_whole_dataframe("crop_batch")

    suite = gx.ExpectationSuite(name="crop_ingestion_suite")
    suite = context.suites.add(suite)

    suite.add_expectation(gx.expectations.ExpectTableColumnsToMatchSet(
        column_set  = list(CONFIG["feature_ranges"].keys()) + ["label"],
        exact_match = True
    ))
    for col in list(CONFIG["feature_ranges"].keys()) + ["label"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=2200,
            max_value=None
        )
    )
    for feature, (lo, hi) in CONFIG["feature_ranges"].items():
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
            column=feature, min_value=lo, max_value=hi, mostly=1.0
        ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="label", value_set=CONFIG["known_labels"]
    ))
    suite.add_expectation(gx.expectations.ExpectColumnDistinctValuesToEqualSet(
        column="label", value_set=CONFIG["known_labels"]
    ))

    val_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name  = "crop_ingestion_validation",
            data  = batch_def,
            suite = suite
        )
    )
    results = val_def.run(batch_parameters={"dataframe": df})

    passed = sum(1 for r in results.results if r.success)
    total  = len(results.results)
    logger.info(f"GE validation: {passed}/{total} expectations passed.")

    if not results.success:
        failed = [
            type(r.expectation_config).__name__
            for r in results.results if not r.success
        ]
        raise ValueError(
            f"GE validation failed — ingestion aborted. Failed: {failed}"
        )
    logger.info("Validation passed — proceeding to ingestion.")

    # Deduplication
    ingestion_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df = df.copy()

    hash_cols     = list(CONFIG["feature_ranges"].keys()) + ["label"]
    df["row_hash"] = df[hash_cols].apply(
        lambda row: hashlib.sha256(
            "|".join(str(v) for v in row).encode()
        ).hexdigest(),
        axis=1
    )

    try:
        with db_engine.connect() as conn:
            existing = {
                row[0] for row in conn.execute(
                    text(f"SELECT row_hash FROM {CONFIG['obt_table']}")
                )
            }
    except Exception as e:
        logger.warning(f"Could not fetch existing hashes: {e}")
        existing = set()

    try:
        with db_engine.connect() as conn:
            max_id_result = conn.execute(
                text(f"SELECT COALESCE(MAX(id), 0) FROM {CONFIG['obt_table']}")
            )
            max_id = int(max_id_result.fetchone()[0])
    except Exception:
        max_id = 0

    df.insert(0, "id", range(max_id + 1, max_id + len(df) + 1))
    df["data_source"]         = CONFIG["data_source"]
    df["ingestion_timestamp"] = ingestion_ts

    obt_cols = [
        "id", "nitrogen", "phosphorus", "potassium",
        "temperature", "humidity", "ph", "rainfall",
        "label", "data_source", "ingestion_timestamp", "row_hash"
    ]
    new_rows = df[~df["row_hash"].isin(existing)][obt_cols]
    logger.info(
        f"CSV: {len(df)} | DB: {len(existing)} | New: {len(new_rows)}"
    )

    if new_rows.empty:
        logger.info("No new rows — ingestion is a no-op.")
        return

    os.makedirs(CONFIG["staging_dir"], exist_ok=True)
    staging_file   = os.path.join(CONFIG["staging_dir"], "crop_obt.csv")
    container_path = "/tmp/crop_obt.csv"

    new_rows.to_csv(
        staging_file, sep="|", index=False,
        header=False, float_format="%.4f"
    )

    try:
        subprocess.run(
            ["docker", "cp", staging_file, f"{container}:{container_path}"],
            check=True, timeout=60
        )
        result = subprocess.run(
            [
                "docker", "exec", container, "bash", "-c",
                f"cpimport -s '|' -n 1 -e 0 "
                f"{CONFIG['database_name']} {CONFIG['obt_table']} {container_path}"
            ],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"cpimport failed:\nstderr: {result.stderr}\nstdout: {result.stdout}")

        _stdout = result.stdout or ""
        logger.info(f"cpimport stdout: {_stdout.strip()}")
        import re as _re
        _rejected = _re.search(r"(\d+)\s+rows?\s+rejected", _stdout, _re.IGNORECASE)
        if _rejected and int(_rejected.group(1)) > 0:
            raise RuntimeError(
                f"cpimport reported {_rejected.group(1)} rejected rows. "
                f"Full output: {_stdout.strip()}"
            )
        logger.info(f"cpimport complete — {len(new_rows)} rows loaded.")
    finally:

        if os.path.exists(staging_file):
            os.remove(staging_file)
        subprocess.run(
            ["docker", "exec", container, "bash", "-c",
             f"rm -f {container_path}"],
            check=False, timeout=15
        )


# TASK GROUP 2 — DATA PREPROCESSING

def feature_engineering():
    import pickle
    import redis
    import pyarrow as pa
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing   import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sqlalchemy              import create_engine

    _, db_url = _get_db_urls()
    db_engine    = create_engine(db_url, pool_pre_ping=True)
    feature_cols = CONFIG["feature_cols"]
    target_col   = "label"

    logger.info("Loading OBT from ColumnStore...")
    df = pd.read_sql(
        f"SELECT {', '.join(feature_cols + [target_col])} "
        f"FROM {CONFIG['obt_table']} ORDER BY id",
        db_engine
    )
    logger.info(f"Loaded {len(df)} rows.")

    X = df[feature_cols].values
    y = df[target_col].values

    label_encoder = LabelEncoder()
    y_encoded     = label_encoder.fit_transform(y)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size    = CONFIG["test_size"],
        random_state = CONFIG["random_state"],
        stratify     = y_encoded
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    logger.info(
        f"Split — train: {len(X_train)}, test: {len(X_test)}"
    )

    def to_arrow_bytes(array):
        if array.ndim == 1:
            df_tmp = pd.DataFrame(array, columns=["target"])
        else:
            df_tmp = pd.DataFrame(array)
        buf    = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(buf, pa.Schema.from_pandas(df_tmp))
        writer.write_table(pa.Table.from_pandas(df_tmp))
        writer.close()
        return buf.getvalue().to_pybytes()

    r = redis.Redis(
        host=CONFIG["redis_host"], port=CONFIG["redis_port"],
        decode_responses=False, socket_timeout=10
    )
    ttl_training  = CONFIG["redis_ttl_training"]
    ttl_artifacts = CONFIG["redis_ttl_artifacts"]

    r.setex("crop:X_train",       ttl_training,  to_arrow_bytes(X_train))
    r.setex("crop:X_test",        ttl_training,  to_arrow_bytes(X_test))
    r.setex("crop:y_train",       ttl_training,  to_arrow_bytes(y_train))
    r.setex("crop:y_test",        ttl_training,  to_arrow_bytes(y_test))
    r.setex("crop:scaler",        ttl_artifacts, pickle.dumps(scaler))
    r.setex("crop:label_encoder", ttl_artifacts, pickle.dumps(label_encoder))

    expected_train = int(len(df) * (1 - CONFIG["test_size"]))
    expected_test  = len(df) - expected_train
    assert X_train.shape == (expected_train, len(feature_cols)), (
        f"Unexpected X_train shape: {X_train.shape}. Expected ({expected_train}, {len(feature_cols)})."
    )
    assert X_test.shape == (expected_test, len(feature_cols)), (
        f"Unexpected X_test shape: {X_test.shape}. Expected ({expected_test}, {len(feature_cols)})."
    )
    assert len(np.unique(y_train)) == 22, (
        f"Expected 22 classes in y_train, found {len(np.unique(y_train))}."
    )
    assert len(np.unique(y_test)) == 22, (
        f"Expected 22 classes in y_test, found {len(np.unique(y_test))}."
    )

    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    assert np.allclose(means, 0, atol=0.1), (
        f"Scaled features not zero-mean. Means: {means.round(4)}"
    )
    assert np.allclose(stds, 1, atol=0.1), (
        f"Scaled features not unit variance. Stds: {stds.round(4)}"
    )

    train_counts = np.bincount(y_train.astype(int))
    test_counts  = np.bincount(y_test.astype(int))
    assert train_counts.min() > 0, (
        f"Stratification failed — at least one class missing from y_train. "
        f"Counts: {train_counts}"
    )
    assert test_counts.min() > 0, (
        f"Stratification failed — at least one class missing from y_test. "
        f"Counts: {test_counts}"
    )

    _train_ratio = train_counts.max() / max(train_counts.min(), 1)
    _test_ratio  = test_counts.max()  / max(test_counts.min(),  1)
    assert _train_ratio < 2.0, (
        f"Train set severely imbalanced after split. Max/min ratio: {_train_ratio:.2f}. "
        f"Counts: {train_counts}"
    )
    assert _test_ratio < 2.0, (
        f"Test set severely imbalanced after split. Max/min ratio: {_test_ratio:.2f}. "
        f"Counts: {test_counts}"
    )

    logger.info("Post-preprocessing validation passed.")
    logger.info(
        f"Training arrays cached (TTL: {ttl_training}s). "
        f"Scaler and encoder cached (TTL: {ttl_artifacts}s)."
    )


# TASK GROUP 3 — MODEL TRAINING

def train_and_register():
    import os
    import pickle
    import time
    import redis
    import pyarrow as pa
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.sklearn
    import shap
    from sklearn.ensemble        import (
        RandomForestClassifier,
        GradientBoostingClassifier
    )
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_validate,
        learning_curve
    )
    from sklearn.metrics         import accuracy_score, f1_score,  confusion_matrix, ConfusionMatrixDisplay
    from mlflow.models.signature import infer_signature
    from mlflow                  import MlflowClient

    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])

    ARTIFACT_DIR = CONFIG["artifact_dir"]
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Restore from Redis
    def from_arrow_bytes(data):
        reader = pa.ipc.open_stream(pa.py_buffer(data))
        return reader.read_all().to_pandas().values.squeeze()

    r = redis.Redis(
        host=CONFIG["redis_host"], port=CONFIG["redis_port"],
        decode_responses=False, socket_timeout=10
    )

    _required_keys = ["crop:X_train", "crop:X_test", "crop:y_train",
                      "crop:y_test", "crop:label_encoder"]
    _missing_keys  = [k for k in _required_keys if r.get(k) is None]
    if _missing_keys:
        raise RuntimeError(
            f"Required Redis keys are missing or expired: {_missing_keys}. "
            f"Re-run the feature_engineering task to repopulate them. "
            f"TTL for training arrays is {CONFIG['redis_ttl_training']}s "
            f"({CONFIG['redis_ttl_training'] // 3600}h)."
        )

    X_train       = from_arrow_bytes(r.get("crop:X_train"))
    X_test        = from_arrow_bytes(r.get("crop:X_test"))
    y_train       = from_arrow_bytes(r.get("crop:y_train"))
    y_test        = from_arrow_bytes(r.get("crop:y_test"))
    label_encoder = pickle.loads(r.get("crop:label_encoder"))

    logger.info(
        f"Restored from Redis — "
        f"X_train: {X_train.shape}, X_test: {X_test.shape}"
    )

    feature_cols = CONFIG["feature_cols"]
    cv = StratifiedKFold(
        n_splits     = CONFIG["cv_folds"],
        shuffle      = True,
        random_state = CONFIG["random_state"]
    )

    candidates = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators = 200,
                max_features = "sqrt",
                random_state = CONFIG["random_state"],
                n_jobs       = -1
            ),
            {"n_estimators": 200, "max_features": "sqrt"}
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators  = 200,
                learning_rate = 0.1,
                max_depth     = 5,
                subsample     = 0.8,
                random_state  = CONFIG["random_state"]
            ),
            {
                "n_estimators" : 200,
                "learning_rate": 0.1,
                "max_depth"    : 5,
                "subsample"    : 0.8
            }
        ),
    ]

    best_run_id   = None
    best_f1       = -1.0
    best_name     = None
    best_model    = None

    for model_name, model, params in candidates:
        logger.info(f"Training {model_name}...")

        with mlflow.start_run(run_name=model_name) as run:

            mlflow.log_param("model_name",   model_name)
            mlflow.log_param("cv_folds",     CONFIG["cv_folds"])
            mlflow.log_param("random_state", CONFIG["random_state"])
            mlflow.log_param("n_features",   X_train.shape[1])
            mlflow.log_param("n_train_rows", X_train.shape[0])
            for k, v in params.items():
                mlflow.log_param(k, v)

            cv_results = cross_validate(
                model, X_train, y_train,
                cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1
            )
            cv_acc_mean = cv_results["test_accuracy"].mean()
            cv_f1_mean  = cv_results["test_f1_macro"].mean()
            cv_f1_std   = cv_results["test_f1_macro"].std()

            mlflow.log_metric("cv_accuracy_mean", round(cv_acc_mean, 4))
            mlflow.log_metric("cv_f1_macro_mean", round(cv_f1_mean,  4))
            mlflow.log_metric("cv_f1_macro_std",  round(cv_f1_std,   4))

            model.fit(X_train, y_train)
            y_pred_test  = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            test_f1  = f1_score(y_test, y_pred_test, average="macro")
            test_acc = accuracy_score(y_test, y_pred_test)

            mlflow.log_metric("test_f1_macro", round(test_f1,  4))
            mlflow.log_metric("test_accuracy", round(test_acc, 4))

            logger.info(
                f"{model_name} — cv_f1: {cv_f1_mean:.4f}, "
                f"test_f1: {test_f1:.4f}, test_acc: {test_acc:.4f}"
            )

            # Confusion matrix — test set
            class_names = label_encoder.classes_
            cm_test = confusion_matrix(y_test, y_pred_test)
            fig, ax = plt.subplots(figsize=(14, 12))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
            disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
            ax.set_title(f"Confusion Matrix (Test Set) — {model_name}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            cm_test_path = os.path.join(ARTIFACT_DIR, f"confusion_matrix_test_{model_name.lower()}.png")
            plt.savefig(cm_test_path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            mlflow.log_artifact(cm_test_path)
            logger.info(f"Test confusion matrix logged: {cm_test_path}")

            # Confusion matrix — train set
            cm_train = confusion_matrix(y_train, y_pred_train)
            fig, ax = plt.subplots(figsize=(14, 12))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
            disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Greens")
            ax.set_title(f"Confusion Matrix (Train Set) — {model_name}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            cm_train_path = os.path.join(ARTIFACT_DIR, f"confusion_matrix_train_{model_name.lower()}.png")
            plt.savefig(cm_train_path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            mlflow.log_artifact(cm_train_path)
            logger.info(f"Train confusion matrix logged: {cm_train_path}")

            logger.info(f"Computing learning curves for {model_name}...")
            train_sizes_rel = np.linspace(0.1, 1.0, 8)

            train_sizes_abs, train_scores, val_scores = learning_curve(
                estimator    = model,
                X            = X_train,
                y            = y_train,
                cv           = cv,
                train_sizes  = train_sizes_rel,
                scoring      = "f1_macro",
                n_jobs       = -1,
                shuffle      = True,
                random_state = CONFIG["random_state"]
            )

            train_mean = train_scores.mean(axis=1)
            train_std  = train_scores.std(axis=1)
            val_mean   = val_scores.mean(axis=1)
            val_std    = val_scores.std(axis=1)
            final_gap  = float(train_mean[-1] - val_mean[-1])

            mlflow.log_metric(
                f"learning_curve_final_gap_{model_name.lower()}",
                round(final_gap, 4)
            )
            mlflow.log_metric(
                f"learning_curve_final_val_f1_{model_name.lower()}",
                round(float(val_mean[-1]), 4)
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_sizes_abs, train_mean, "o-",
                    color="#2196F3", linewidth=2, label="Training F1 Macro")
            ax.fill_between(train_sizes_abs,
                            train_mean - train_std,
                            train_mean + train_std,
                            alpha=0.15, color="#2196F3")
            ax.plot(train_sizes_abs, val_mean, "s-",
                    color="#FF5722", linewidth=2,
                    label="Validation F1 Macro (CV)")
            ax.fill_between(train_sizes_abs,
                            val_mean - val_std,
                            val_mean + val_std,
                            alpha=0.15, color="#FF5722")
            ax.annotate(
                f"Gap at full size: {final_gap:.4f}",
                xy        = (train_sizes_abs[-1], val_mean[-1]),
                xytext    = (train_sizes_abs[-2], val_mean[-1] - 0.02),
                fontsize  = 9,
                arrowprops= dict(arrowstyle="->", color="black", lw=1)
            )
            ax.set_title(f"Learning Curves — {model_name}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Training Set Size", fontsize=11)
            ax.set_ylabel("F1 Macro Score",    fontsize=11)
            ax.set_ylim(0.85, 1.02)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            lc_path = os.path.join(
                ARTIFACT_DIR,
                f"learning_curve_{model_name.lower()}.png"
            )
            plt.savefig(lc_path, bbox_inches="tight", dpi=120)
            plt.close()
            mlflow.log_artifact(lc_path)
            logger.info(f"Learning curve logged: {lc_path}")

            signature = infer_signature(X_train, y_pred_test)
            mlflow.sklearn.log_model(
                sk_model      = model,
                artifact_path = "model",
                signature     = signature,
                input_example = X_train[:3]
            )

            if test_f1 > best_f1:
                best_f1    = test_f1
                best_run_id = run.info.run_id
                best_name  = model_name
                best_model = model

    MIN_ACCEPTABLE_F1 = 0.90
    if best_f1 < MIN_ACCEPTABLE_F1:
        raise ValueError(
            f"Best model '{best_name}' has test F1 {best_f1:.4f}, "
            f"which is below the minimum acceptable threshold of {MIN_ACCEPTABLE_F1}. "
            f"Refusing to promote to Production. "
            f"Check the data pipeline for issues before retraining."
        )
    logger.info(f"Quality gate passed — F1 {best_f1:.4f} >= {MIN_ACCEPTABLE_F1}.")

    _scaler_raw = r.get("crop:scaler")
    if _scaler_raw:
        _scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
        with open(_scaler_path, "wb") as _f:
            _f.write(_scaler_raw)
        with mlflow.start_run(run_id=best_run_id):
            mlflow.log_artifact(_scaler_path, artifact_path="")
        logger.info(f"Scaler logged to MLflow run {best_run_id} as scaler.pkl.")
    else:
        logger.warning(
            "crop:scaler not in Redis — scaler.pkl not logged to MLflow. "
            "Monitoring fallback will fail if Redis expires."
        )

    # SHAP analysis on best model only
    
    logger.info(
        f"Running SHAP analysis on best model: {best_name} "
        f"(test_f1: {best_f1:.4f})"
    )

    with mlflow.start_run(run_name=f"shap_{best_name}"):
        mlflow.log_param("model_explained", best_name)
        mlflow.log_param("best_run_id",     best_run_id)
        mlflow.log_param("best_test_f1",    round(best_f1, 4))

        explainer   = shap.TreeExplainer(best_model)
        shap_values = np.array(explainer.shap_values(X_test))

        if shap_values.ndim == 3 and shap_values.shape[0] == X_test.shape[0]:
            shap_values_3d = shap_values.transpose(2, 0, 1)
        elif shap_values.ndim == 3 and shap_values.shape[2] == X_test.shape[0]:
            shap_values_3d = shap_values.transpose(0, 2, 1)
        else:
            shap_values_3d = shap_values

        mean_abs_shap = np.abs(shap_values_3d).mean(axis=(0, 1))

        top_idx     = int(np.argmax(mean_abs_shap))
        top_feature = feature_cols[top_idx]
        mlflow.log_param("shap_top_feature", top_feature)
        mlflow.log_metric(
            "shap_top_feature_importance",
            round(float(mean_abs_shap[top_idx]), 4)
        )
        for feat, val in zip(feature_cols, mean_abs_shap):
            mlflow.log_metric(f"shap_{feat}", round(float(val), 4))

        # Global importance bar plot
        importance_df = pd.DataFrame({
            "feature"   : feature_cols,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(
            importance_df["feature"],
            importance_df["importance"],
            color="#4C72B0", edgecolor="white"
        )
        for bar, val in zip(bars, importance_df["importance"]):
            ax.text(
                val + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9
            )
        ax.set_xlabel("Mean |SHAP Value| across all classes", fontsize=11)
        ax.set_title(
            f"Global Feature Importance — {best_name} (Best Model)",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        shap_bar_path = os.path.join(
            ARTIFACT_DIR,
            f"shap_global_importance_{best_name.lower()}.png"
        )
        plt.savefig(shap_bar_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        mlflow.log_artifact(shap_bar_path)

        # SHAP vs Gini comparison
        gini_importance = best_model.feature_importances_
        comparison_df   = pd.DataFrame({
            "feature"         : feature_cols,
            "shap_importance" : mean_abs_shap,
            "gini_importance" : gini_importance
        }).sort_values("shap_importance", ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"SHAP vs Gini Importance — {best_name} (Best Model)",
            fontsize=13, fontweight="bold"
        )
        axes[0].barh(
            comparison_df["feature"][::-1],
            comparison_df["shap_importance"][::-1],
            color="#4C72B0"
        )
        axes[0].set_title("Mean |SHAP Value|")
        axes[0].set_xlabel("Importance")

        axes[1].barh(
            comparison_df["feature"][::-1],
            comparison_df["gini_importance"][::-1],
            color="#DD8452"
        )
        axes[1].set_title("Gini Importance")
        axes[1].set_xlabel("Importance")
        plt.tight_layout()

        shap_comp_path = os.path.join(
            ARTIFACT_DIR,
            f"shap_vs_gini_{best_name.lower()}.png"
        )
        plt.savefig(shap_comp_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        mlflow.log_artifact(shap_comp_path)

        logger.info(
            f"SHAP complete — top feature: {top_feature} "
            f"({mean_abs_shap[top_idx]:.4f})"
        )

    # Register best model to Production

    logger.info(
        f"Registering best model: {best_name} "
        f"(test_f1: {best_f1:.4f}, run_id: {best_run_id})"
    )

    client    = MlflowClient(tracking_uri=CONFIG["mlflow_tracking_uri"])
    model_uri = f"runs:/{best_run_id}/model"

    registered = mlflow.register_model(
        model_uri = model_uri,
        name      = CONFIG["registered_model_name"]
    )

    _ready = False
    for _attempt in range(10):
        details = client.get_model_version(
            name    = CONFIG["registered_model_name"],
            version = registered.version
        )
        if details.status == "READY":
            _ready = True
            break
        logger.info(
            f"Waiting for model version {registered.version} to become READY "
            f"(attempt {_attempt + 1}/10, current status: {details.status})..."
        )
        time.sleep(2)

    if not _ready:
        raise RuntimeError(
            f"Model version {registered.version} did not reach READY status "
            f"after 20 seconds. Current status: {details.status}. "
            f"Check MLflow server logs."
        )

    try:
        client.set_registered_model_alias(
            name    = CONFIG["registered_model_name"],
            alias   = "Production",
            version = registered.version
        )
        logger.info(
            f"Model version {registered.version} aliased as 'Production'."
        )
        logger.info(
            f"Model version {registered.version} promoted to Production."
        )
        for key in ["crop:X_train", "crop:X_test",
                    "crop:y_train", "crop:y_test"]:
            r.delete(key)
        logger.info("Redis training cache cleaned up.")
    except Exception as e:
        logger.error(
            f"Model promotion failed: {e}. "
            f"Redis cache preserved for retry."
        )
        raise


# DAG DEFINITION

with DAG(
    dag_id            = "crop_recommendation_pipeline",
    default_args      = default_args,
    start_date        = pdatetime(2024, 1, 1, tz="UTC"),
    schedule_interval = "@daily",
    catchup           = False,
    tags              = ["crop", "columnstore", "mlops"],
    doc_md            = """
## Crop Recommendation System — MLOps Pipeline

End-to-end pipeline structured into four TaskGroups.

**TaskGroup 1 — Data Ingestion**
- Check and start ColumnStore
- Create schema (idempotent, with commit)
- GE validation gate + cpimport to OBT

**TaskGroup 2 — Data Preprocessing**
- Split first, then StandardScaler on train only (no data leakage)
- LabelEncoder + stratified split
- Cache to Redis (PyArrow + pickle)

**TaskGroup 3 — Model Training**
- Train RandomForest and GradientBoosting
- learning_curve runs after model.fit()
- Log to MLflow experiment
- Register best model to Production stage
- Clean up Redis only after confirmed promotion
    """
) as dag:

    with TaskGroup(
        group_id = "task_group_1_data_ingestion",
        tooltip  = "ColumnStore startup, schema creation, GE validation, OBT ingestion"
    ) as tg1:
        t1 = PythonOperator(
            task_id         = "check_and_start_columnstore",
            python_callable = check_and_start_columnstore,
        )
        t2 = PythonOperator(
            task_id         = "create_schema",
            python_callable = create_schema,
        )
        t3 = PythonOperator(
            task_id         = "validate_and_ingest",
            python_callable = validate_and_ingest,
        )
        t1 >> t2 >> t3

    with TaskGroup(
        group_id = "task_group_2_data_preprocessing",
        tooltip  = "Feature scaling, label encoding, train-test split, Redis caching"
    ) as tg2:
        t4 = PythonOperator(
            task_id         = "feature_engineering",
            python_callable = feature_engineering,
        )

    with TaskGroup(
        group_id = "task_group_3_model_training",
        tooltip  = "Model training, MLflow tracking, model registration, Redis cleanup"
    ) as tg3:
        t5 = PythonOperator(
            task_id         = "train_and_register",
            python_callable = train_and_register,
        )

    tg1 >> tg2 >> tg3
