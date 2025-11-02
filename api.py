from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = FastAPI(title="IRIS Classifier API")

# Load the best model

# Setting the experiment and the tracking uri
public_ip = '136.111.227.110'
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
    ! dvc checkout
    ! python train_model.py augmented_train_v2
    return {"message":"Model initialized and trained successfully!"}

@app.post("/predict")
def predict_species(data: IrisInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {
        "predicted_class": prediction
    }

    
