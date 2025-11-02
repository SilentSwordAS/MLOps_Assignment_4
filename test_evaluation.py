import pytest
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Setting the experiment and the tracking uri
public_ip = '34.9.255.250'
mlflow.set_tracking_uri(f"http://{public_ip}:7600/")
mlflow.set_experiment("Iris_Classifier_Pipeline_2")
client = MlflowClient(tracking_uri = f"http://{public_ip}:7600/")

# Loading the dataset
data = pd.read_csv('augmented_train_v2.csv')
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['species'])
le = LabelEncoder()

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
y_train = le.fit_transform(X_train['species'])

# Test Cases

def test_model_exists():
    assert model_loaded

def test_model_accuracy():
    y_pred = model_loaded.predict(X_test.drop(['species'], axis=1))
    model_score = accuracy_score(le.transform(X_test['species']), y_pred)
    assert model_score > 0.9

def test_data_header_validation():
    data_columns = list(data.columns.values)
    assert len(data_columns) == 5

def test_data_number_validation():
    assert len(data) == 300


