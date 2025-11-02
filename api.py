from fastapi import FastAPI
from pydantic import BaseModel
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

app = FastAPI(title="IRIS Classifier API")

le = LabelEncoder()

if not os.path.isfile('augmented_train_v2.csv'):
    subprocess.run('dvc checkout',shell=True)

data = pd.read_csv('augmented_train_v2.csv')
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['species'])
y_train = le.fit_transform(X_train['species'])
# Load the best model

# Setting the experiment and the tracking uri
public_ip = '136.116.214.14'
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
def predict_species(data: IrisInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model_loaded.predict(input_df)
    return {
        "predicted_class": str(le.inverse_transform(prediction)[0])
    }

@app.get("/train")
def train_model():
    subprocess.run('python train_model.py augmented_train_v2',shell=True)
    return {"message":"Model Trained Successfully!"}
