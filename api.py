import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import subprocess
import os
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import logging
import time
import json

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("iris-classifier-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# Simulated flags, normally these would be set by various parts of the code
# e.g. if model load is taking time due to weights being large,
#  then is_ready would be False until the model is loaded.
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    import time
    time.sleep(2)  # simulate work, normally this would be model loading
    app_state["is_ready"] = True

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

app = FastAPI(title="IRIS Classifier API")

le = LabelEncoder()

if os.path.isfile('augmented_train_v2.csv'):
  subprocess.run('python train_model.py augmented_train_v2',shell=True)

data = pd.read_csv('augmented_train_v2.csv')
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['species'])
y_train = le.fit_transform(X_train['species'])
# Load the best model

# Setting the experiment and the tracking uri
public_ip = '35.188.196.11'
mlflow.set_tracking_uri(f"http://{public_ip}:7600/")
mlflow.set_experiment("Iris_Classifier_Pipeline_2")
client = MlflowClient(tracking_uri = f"http://{public_ip}:7600/")

# Need to get experiment_id to access the run_id and the model name of our best model to register it.

experiment_id = mlflow.get_experiment_by_name("Iris_Classifier_Pipeline_2").experiment_id
runs_df = mlflow.search_runs(experiment_ids=[experiment_id], order_by=[f"metrics.accuracy {'DESC'}"])
runs_df = runs_df[runs_df['status']=='FINISHED']
best_run_id = runs_df.iloc[0]['run_id']
best_run_model_name = runs_df.iloc[0]['tags.mlflow.runName']

# Get all versions of the model

all_versions = client.search_model_versions(f"name='{best_run_model_name}'")

# Loading the Model

model_loaded = mlflow.sklearn.load_model(f"models:/{best_run_model_name}/{max([v.version for v in all_versions])}")

# Input Schema

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get('/')
def read_root():
    return {"message":"Model initialized successfully!"}

@app.post("/predict")
async def predict(input: IrisInput, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = pd.DataFrame([input.dict()])
            result = model_loaded.predict(input_data)
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }))
            return {
                "predicted_class": str(le.inverse_transform(result)[0])
            }

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")

