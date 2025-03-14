import numpy as np
import pytest
import mlflow.sklearn
import os

@pytest.mark.parametrize(
    "input_data",
    [
        np.array([[1, 2014, 145500, 23.4, 1, 0, 0]])  # Ensure input is a 2D array
    ]
)
def test_input_model(input_data):
    expected_shape = (1,)  # Expecting a single output value

    # Load model from MLflow
    mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
    model_uri = "models:/st124856-a3-model/1"  
    sklearn_model = mlflow.sklearn.load_model(model_uri)

    # Make prediction
    selling_price_3 = sklearn_model.predict(input_data)

    # Assert output shape matches expectation
    assert selling_price_3.shape == expected_shape, f"Expected {expected_shape}, got {selling_price_3.shape}"

@pytest.mark.parametrize(
    "input_data, expected_shape",
    [
        (np.array([[1, 2014, 145500, 23.4, 1, 0, 0]]), (1,))  # Single sample, expecting 1 output
    ]
)
def test_output_model(input_data, expected_shape):
    # Load model from MLflow
    mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin" 
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
    model_uri = "models:/st124856-a3-model/1"  
    sklearn_model = mlflow.sklearn.load_model(model_uri)

    # Make prediction
    selling_price_3 = sklearn_model.predict(input_data)

    # Assert output shape matches expectation
    assert selling_price_3.shape == expected_shape, f"Expected {expected_shape}, got {selling_price_3.shape}"
