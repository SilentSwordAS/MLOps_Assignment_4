from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys
import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient

# Setting the experiment and the tracking uri
public_ip = '34.136.156.223'
mlflow.set_tracking_uri(f"http://{public_ip}:7600/")
mlflow.set_experiment("Iris_Classifier_Pipeline_2")

# Defining schemas for ModelSignature
input_schema = Schema([
    ColSpec("float", "sepal_length"),
    ColSpec("float",'sepal_width'),
    ColSpec("float",'petal_length'),
    ColSpec("float",'petal_width')
])

output_schema = Schema([
    ColSpec("string", "species")
])


# Loading the Dataset and initializing the label encoder
data = pd.read_csv(f'{sys.argv[1]}.csv')
le = LabelEncoder()

X_train,X_test = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data['species']
)

# Defining multiple hyperparameters to perform hyperparameter tuning

param_grid = [
    {"criterion":"entropy", "max_depth": 3},
    {"criterion":"log_loss", "max_depth": 4},  # 6
    {"criterion":"gini", "max_depth": 2} # 8
]

for param in param_grid:
    with mlflow.start_run() as run:
        model = DecisionTreeClassifier(**param)
        model.fit(X_train.drop(['species'],axis=1), le.fit_transform(X_train['species']))

        y_test = le.transform(X_test['species'])
        y_pred = model.predict(X_test.drop(['species'],axis=1))

        acc_score = accuracy_score(y_test, y_pred)

        signature = ModelSignature(inputs = input_schema, outputs = output_schema)

        mlflow.log_params(param)
        mlflow.log_metric("accuracy", acc_score)
        mlflow.sklearn.log_model(sk_model=model, signature = signature,name="model")

# Need to get experiment_id to access the run_id and the model name of our best model to register it.

experiment_id = mlflow.get_experiment_by_name("Iris_Classifier_Pipeline_2").experiment_id
runs_df = mlflow.search_runs(experiment_ids=[experiment_id], order_by=[f"metrics.accuracy {'DESC'}"])
runs_df = runs_df[runs_df['status']=='FINISHED']
best_run_id = runs_df.iloc[0]['run_id']
best_run_model_name = runs_df.iloc[0]['tags.mlflow.runName']


# Registering the model

model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri, best_run_model_name)
print(f"Registered model {best_run_model_name} with the following run_id {best_run_id}.")
