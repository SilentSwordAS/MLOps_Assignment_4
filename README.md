### train_model.py -> For Training the model as well as registering the model on MLFlow
### test_evaluation.py -> For predictions using the best model stored in mlflow as well as conducting appropriate unit tests using pytest
### augment_iris.py -> Concatenating datasets 
### .dvc -> Contains the cache required to reconstruct the data from .dvc files
### .github -> Contains the yaml file for CI/CD
### mlartifacts -> Contains the necessary files required for MLFlow to work according to the parameters set currrently.
### DockerFile -> Used for defining what all commands and parameters to set while building the docker container.
### *.dvc -> Files which can be reconstructed into the dataset as long as the cache is available in the .dvc directory.
### *.csv -> CSV files of the dataset used.
