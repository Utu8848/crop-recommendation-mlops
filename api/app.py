import os
import uuid
import pickle
import logging
from datetime     import datetime
from contextlib   import asynccontextmanager
from typing       import Annotated

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from dotenv import load_dotenv

from fastapi            import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses  import JSONResponse
from pydantic           import BaseModel, Field, field_validator, model_validator
from sqlalchemy         import create_engine, text

load_dotenv("/home/utsav/airflow/.env")

# Logging

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("crop_api")

# Configuration

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI",   "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "crop_recommendation_model")

MARIADB_URL = os.getenv("MARIADB_URL")
if not MARIADB_URL:
    raise EnvironmentError(
        "MARIADB_URL environment variable is not set. "
        "Check that /home/utsav/airflow/.env exists, is readable, and "
        "defines MARIADB_URL (see .env.example)."
    )

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

FEATURE_COLS = [
    "nitrogen", "phosphorus", "potassium",
    "temperature", "humidity", "ph", "rainfall"
]

# Module-level state — loaded once at startup, reused across all requests

MODEL         = None
SCALER        = None
LABEL_ENCODER = None
MODEL_VERSION = None
DB_ENGINE     = None


# Startup / shutdown lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, SCALER, LABEL_ENCODER, MODEL_VERSION, DB_ENGINE

    logger.info("Starting up — loading model and artifacts from MLflow...")

    # MLflow client
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    try:
        version_info  = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, "Production"
        )
    except Exception:
        raise RuntimeError(
            f"No 'Production' alias found for model '{REGISTERED_MODEL_NAME}'. "
            "Run the Airflow pipeline (train_and_register task) first."
        )

    MODEL_VERSION = f"v{version_info.version}_run_{version_info.run_id[:8]}"
    model_uri     = f"models:/{REGISTERED_MODEL_NAME}@Production"
    MODEL         = mlflow.sklearn.load_model(model_uri)
    logger.info(
        f"Model loaded: {REGISTERED_MODEL_NAME} "
        f"version {version_info.version} (run {version_info.run_id[:8]})"
    )

    # Scaler — MLflow artifact first, Redis fallback
    scaler_loaded = False
    try:
        mlflow.artifacts.download_artifacts(
            run_id        = version_info.run_id,
            artifact_path = "",
            dst_path      = "/tmp/crop_artifacts"
        )
        for root, _, files in os.walk("/tmp/crop_artifacts"):
            for fname in files:
                if fname.endswith(".pkl") and "scaler" in fname:
                    with open(os.path.join(root, fname), "rb") as f:
                        SCALER = pickle.load(f)
                    logger.info(f"Scaler loaded from MLflow artifact: {fname}")
                    scaler_loaded = True
                    break
            if scaler_loaded:
                break
    except Exception as e:
        logger.warning(f"MLflow artifact download failed: {e}")

    if not scaler_loaded:
        logger.warning("Scaler not found in MLflow artifacts — falling back to Redis.")
        import redis as redis_lib
        r      = redis_lib.Redis(host="127.0.0.1", port=6379, decode_responses=False)
        raw    = r.get("crop:scaler")
        if raw is None:
            raise RuntimeError(
                "Scaler not found in MLflow artifacts or Redis. "
                "Re-run the feature_engineering and train_and_register tasks."
            )
        SCALER = pickle.loads(raw)
        logger.info("Scaler loaded from Redis fallback.")

    # Label encoder — Redis
    import redis as redis_lib
    r   = redis_lib.Redis(host="127.0.0.1", port=6379, decode_responses=False)
    raw = r.get("crop:label_encoder")
    if raw is None:
        raise RuntimeError(
            "crop:label_encoder not found in Redis (TTL may have expired). "
            "Re-run the feature_engineering task to repopulate it. "
            "Consider also saving it as an MLflow artifact in train_and_register."
        )
    LABEL_ENCODER = pickle.loads(raw)
    logger.info(f"Label encoder loaded. Classes: {list(LABEL_ENCODER.classes_)}")

    # Database engine
    DB_ENGINE = create_engine(MARIADB_URL, pool_pre_ping=True, pool_size=5)
    logger.info("Database engine initialised.")

    logger.info("Startup complete — API ready.")
    yield

    # Shutdown
    logger.info("Shutting down — disposing DB engine.")
    if DB_ENGINE:
        DB_ENGINE.dispose()


# FastAPI app

app = FastAPI(
    title       = "Crop Recommendation API",
    description = "MLOps pipeline serving endpoint for crop recommendation.",
    version     = "1.0.0",
    lifespan    = lifespan
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Log Pydantic validation rejections before returning HTTP 422.
    Rejected inputs never reach the model or predictions_log, which protects
    the monitoring baseline used by Task Group 4 (Evidently drift detection).
    """
    logger.warning(f"Input validation rejected — errors: {exc.errors()}")
    return JSONResponse(
        status_code = 422,
        content     = {"detail": exc.errors()}
    )


# PYDANTIC SCHEMAS

class CropPredictionRequest(BaseModel):
    nitrogen    : Annotated[float, Field(..., ge=0,    le=140,   description="Nitrogen (kg/ha)")]
    phosphorus  : Annotated[float, Field(..., ge=5,    le=145,   description="Phosphorus (kg/ha)")]
    potassium   : Annotated[float, Field(..., ge=5,    le=205,   description="Potassium (kg/ha)")]
    temperature : Annotated[float, Field(..., ge=0.0,  le=50.0,  description="Temperature (°C)")]
    humidity    : Annotated[float, Field(..., ge=14.0, le=100.0, description="Relative humidity (%)")]
    ph          : Annotated[float, Field(..., ge=3.5,  le=10.0,  description="Soil pH")]
    rainfall    : Annotated[float, Field(..., ge=20.0, le=3000.0,description="Annual rainfall (mm)")]

    model_config = {"json_schema_extra": {
        "example": {
            "nitrogen"    : 90,
            "phosphorus"  : 42,
            "potassium"   : 43,
            "temperature" : 20.88,
            "humidity"    : 82.00,
            "ph"          : 6.50,
            "rainfall"    : 202.94
        }
    }}

    @field_validator("ph")
    @classmethod
    def ph_precision(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError(
                f"ph={v} is below any physically possible soil value. "
                "Check your sensor or input — soil pH cannot be below 1.0."
            )
        return round(v, 4)

    @model_validator(mode="after")
    def no_all_zero_nutrients(self) -> "CropPredictionRequest":
        if self.nitrogen == 0 and self.phosphorus == 0 and self.potassium == 0:
            raise ValueError(
                "nitrogen, phosphorus, and potassium cannot all be zero — "
                "this indicates a missing or corrupt input payload."
            )
        return self


class CropPredictionResponse(BaseModel):
    request_id       : str
    predicted_crop   : str
    confidence_score : Annotated[float, Field(ge=0.0, le=1.0)]
    model_version    : str
    timestamp        : str


# Helper — write to predictions_log (InnoDB)

def log_prediction(
    request_id       : str,
    input_data       : CropPredictionRequest,
    predicted_crop   : str,
    confidence_score : float,
    response_status  : str
):
    try:
        with DB_ENGINE.connect() as conn:
            conn.execute(text("""
                INSERT INTO predictions_log (
                    request_id, nitrogen, phosphorus, potassium,
                    temperature, humidity, ph, rainfall,
                    predicted_crop, confidence_score,
                    model_version, prediction_timestamp, response_status
                ) VALUES (
                    :request_id, :nitrogen, :phosphorus, :potassium,
                    :temperature, :humidity, :ph, :rainfall,
                    :predicted_crop, :confidence_score,
                    :model_version, :prediction_timestamp, :response_status
                )
            """), {
                "request_id"          : request_id,
                "nitrogen"            : input_data.nitrogen,
                "phosphorus"          : input_data.phosphorus,
                "potassium"           : input_data.potassium,
                "temperature"         : input_data.temperature,
                "humidity"            : input_data.humidity,
                "ph"                  : input_data.ph,
                "rainfall"            : input_data.rainfall,
                "predicted_crop"      : predicted_crop,
                "confidence_score"    : round(confidence_score, 4),
                "model_version"       : MODEL_VERSION,
                "prediction_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "response_status"     : response_status,
            })
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log prediction to DB [{request_id}]: {e}")


# POST /predict

@app.post("/predict", response_model=CropPredictionResponse)
def predict(request: CropPredictionRequest):
    """
    Predict the recommended crop given soil and climate measurements.
    Pydantic validates the payload before this function body executes —
    invalid inputs return HTTP 422 and are never written to predictions_log.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Prediction request [{request_id}]: {request.model_dump()}")

    try:
        features = np.array([[
            request.nitrogen,
            request.phosphorus,
            request.potassium,
            request.temperature,
            request.humidity,
            request.ph,
            request.rainfall,
        ]])

        features_scaled  = SCALER.transform(features)
        prediction_index = MODEL.predict(features_scaled)[0]
        probabilities    = MODEL.predict_proba(features_scaled)[0]
        confidence_score = float(probabilities[prediction_index])
        predicted_crop   = LABEL_ENCODER.inverse_transform([prediction_index])[0]

        logger.info(
            f"Prediction [{request_id}]: {predicted_crop} "
            f"(confidence: {confidence_score:.4f})"
        )

        log_prediction(
            request_id       = request_id,
            input_data       = request,
            predicted_crop   = predicted_crop,
            confidence_score = confidence_score,
            response_status  = "success"
        )

        return CropPredictionResponse(
            request_id       = request_id,
            predicted_crop   = predicted_crop,
            confidence_score = round(confidence_score, 4),
            model_version    = MODEL_VERSION,
            timestamp        = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        )

    except Exception as e:
        logger.error(f"Prediction failed [{request_id}]: {e}")
        log_prediction(
            request_id       = request_id,
            input_data       = request,
            predicted_crop   = "error",
            confidence_score = 0.0,
            response_status  = "failed"
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# GET /health

@app.get("/health")
def health():
    """
    Liveness + readiness check.
    Returns 503 if MariaDB is unreachable or the model failed to load.
    Used by load balancers and can be polled by an Airflow HttpSensor.
    """
    status = {
        "api"     : "ok",
        "model"   : "not_loaded",
        "database": "unreachable"
    }

    if MODEL is not None:
        status["model"] = f"loaded — {REGISTERED_MODEL_NAME} {MODEL_VERSION}"

    try:
        with DB_ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["database"] = "ok"
    except Exception as e:
        status["database"] = f"error: {str(e)}"

    all_ok = all(
        v == "ok" or (isinstance(v, str) and v.startswith("loaded"))
        for v in status.values()
    )
    if not all_ok:
        raise HTTPException(status_code=503, detail=status)

    return status


# GET /metrics

@app.get("/metrics")
def metrics():
    """
    Lightweight prediction statistics from predictions_log.
    Consumed by Evidently monitoring in Task Group 4 of the Airflow DAG.
    """
    try:
        with DB_ENGINE.connect() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM predictions_log WHERE response_status = 'success'")
            ).fetchone()[0]

            row = conn.execute(text("""
                SELECT
                    MIN(confidence_score)          AS min_confidence,
                    MAX(confidence_score)          AS max_confidence,
                    AVG(confidence_score)          AS avg_confidence,
                    COUNT(DISTINCT predicted_crop) AS unique_crops_predicted
                FROM predictions_log
                WHERE response_status = 'success'
            """)).fetchone()

        return {
            "total_predictions"      : total,
            "min_confidence"         : round(float(row[0]), 4) if row[0] else None,
            "max_confidence"         : round(float(row[1]), 4) if row[1] else None,
            "avg_confidence"         : round(float(row[2]), 4) if row[2] else None,
            "unique_crops_predicted" : row[3],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics query failed: {str(e)}")
