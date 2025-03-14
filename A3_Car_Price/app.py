#Code Here
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import os
# Load the trained model
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# Define model URI
model_uri = "models:/st124856-a3-model/1"

# Load the model from MLflow
model = mlflow.sklearn.load_model(model_uri)

# Initialize the Dash app
app = Dash()

# Layout
app.layout = html.Div(
    children=[
        # Header
        html.H1(children="Car Price Prediction for Jirapon's Company", style={'textAlign': 'center'}),

        # Instructions Section
        html.Div(
            children=[
                html.H2("Instruction"),
                html.P("In order to predict car price, you need to choose Year, Kilometer Driven, Mileage, and Car Brand."),
                html.P("If you don't know the values, default values can help you predict."),
            ],
            style={'marginBottom': '20px'}
        ),

        # Input Fields
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label("Year"),
                        dcc.Input(id="year", type="number", value=2023, style={'width': '100%', 'padding': '8px'}),
                    ],
                    style={'marginBottom': '15px'}
                ),
                html.Div(
                    children=[
                        html.Label("Kilometer Driven (km)"),
                        dcc.Input(id="km_driven", type="number", value=14000, style={'width': '100%', 'padding': '8px'}),
                    ],
                    style={'marginBottom': '15px'}
                ),
                html.Div(
                    children=[
                        html.Label("Mileage (kmpl)"),
                        dcc.Input(id="mileage", type="number", value=20, style={'width': '100%', 'padding': '8px'}),
                    ],
                    style={'marginBottom': '15px'}
                ),
                html.Div(
                    children=[
                        html.Label("Car Brand"),
                        dcc.Dropdown(
                            id="car-brand",
                            options=[
                                {"label": "brand_USA", "value": "USA"},
                                {"label": "brand_European", "value": "European"},
                                {"label": "brand_Asian", "value": "Asian"},
                            ],
                            value="USA",  # Default selected value
                            style={'width': '100%', 'padding': '8px'}
                        ),
                    ],
                    style={'marginBottom': '15px'}
                ),
            ]
        ),

        # Predict Button
        html.Button("predict", id="predict-button", n_clicks=0, style={
            'width': '100%',
            'padding': '10px',
            'backgroundColor': '#007bff',
            'color': 'white',
            'border': 'none',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'fontSize': '16px'
        }),

        # Prediction Result
        html.Div(
            id="result",
            children="The predicted car price will appear here.",
            style={'marginTop': '20px', 'fontWeight': 'bold'}
        ),
    ],
    style={'maxWidth': '600px', 'margin': '0 auto', 'fontFamily': 'Arial, sans-serif'}
)

# Callback for prediction
@app.callback(
    Output("result", "children"),
    [
        Input("predict-button", "n_clicks"),
        Input("year", "value"),
        Input("km_driven", "value"),
        Input("mileage", "value"),
        Input("car-brand", "value"),
    ]
)
def predict_price(n_clicks, year, km_driven, mileage, car_brand):
    if n_clicks > 0:  # Check if the button was clicked
        try:
            if year is None or km_driven is None or mileage is None or car_brand is None:
                return "Please fill in all fields before predicting."

            # One-hot encode car brands with the correct column names
            brand_USA = 1 if car_brand == "USA" else 0
            brand_European = 1 if car_brand == "European" else 0
            brand_Asian = 1 if car_brand == "Asian" else 0

            # Create input DataFrame with corrected column names
            input_data = pd.DataFrame({
                "year": [year],
                "km_driven": [km_driven],
                "mileage": [mileage],
                "brand_USA": [brand_USA],
                "brand_European": [brand_European],
                "brand_Asian": [brand_Asian]
            })

            # Log the input data for debugging
            print(f"Input DataFrame for prediction:\n{input_data}")

            # Predict the price
            predicted_price = model.predict(input_data)[0]
            predicted_price = np.exp(predicted_price)

            return f"The predicted car price is = {predicted_price:,.2f} Baht"
        except Exception as e:
            print(f"Error during prediction: {e}")
            return f"An error occurred: {e}"
    return "Click the predict button to see the result."


if __name__ == "__main__":
    app.run_server(debug=True)