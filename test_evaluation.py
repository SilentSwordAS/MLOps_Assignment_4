import pytest
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('augmented_train_v2.csv')
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['species'])
le = LabelEncoder()

y_train = le.fit_transform(X_train['species'])

def test_model_exists():
    model = joblib.load('augmented_train_v2_model.joblib')
    assert model

def test_model_accuracy():
    model = joblib.load('augmented_train_v2_model.joblib')
    y_pred = model.predict(X_test.drop(['species'], axis=1))
    model_score = accuracy_score(le.transform(X_test['species']), y_pred)
    assert model_score > 0.9

def test_data_header_validation():
    data_columns = list(data.columns.values)
    assert len(data_columns) == 5

def test_data_number_validation():
    assert len(data) == 300


