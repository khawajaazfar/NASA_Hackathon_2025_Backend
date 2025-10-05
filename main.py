import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os

# --- 1. Load the Model ---
MODEL_PATH = "xgboost_air_quality_model.joblib"

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}. "
        "Please ensure 'xgboost_air_quality_model.joblib' is in the same directory."
    )

try:
    # Load the trained model using joblib
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")


# --- 2. Define the FastAPI Application ---
app = FastAPI(
    title="Air Quality Prediction API",
    description="API for predicting 6 air pollutants based on Latitude and Longitude using an XGBoost Multi-Output Regressor.",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- 3. Define Input Data Structure (Pydantic Schema) ---
# This ensures that the incoming JSON data is valid
class PredictionInput(BaseModel):
    """Schema for a single location input."""
    Latitude: float = Field(..., description="Geographic Latitude of the monitoring station.")
    Longitude: float = Field(..., description="Geographic Longitude of the monitoring station.")

class PredictionOutput(BaseModel):
    """Schema for a single prediction output."""
    PM2_5: float = Field(..., description="Predicted PM2.5 concentration (ug/m3).")
    PM10: float = Field(..., description="Predicted PM10 concentration (ug/m3).")
    O3: float = Field(..., description="Predicted Ozone concentration (ppb).")
    NO2: float = Field(..., description="Predicted Nitrogen Dioxide concentration (ppb).")
    CO: float = Field(..., description="Predicted Carbon Monoxide concentration (ppm).")
    SO2: float = Field(..., description="Predicted Sulfur Dioxide concentration (ppb).")

# The endpoint expects a list of these inputs
class PredictionBatchInput(BaseModel):
    """Schema for batch prediction input (a list of locations)."""
    locations: List[PredictionInput]


# --- 4. Define API Endpoints ---

# @app.get("/")
# def read_root():
#     """Returns a simple status message for the root endpoint."""
#     return {"message": "Air Quality Prediction API is running. Go to /docs for the interactive API documentation."}


@app.post("/predict", response_model=List[PredictionOutput])
async def predict_air_quality(data: PredictionBatchInput):
    """
    Accepts a list of Latitude/Longitude pairs and returns the predicted
    concentrations for 6 air pollutants for each location.
    """
    try:
        # Convert list of Pydantic models to a DataFrame
        input_data = [loc.dict() for loc in data.locations]
        X_df = pd.DataFrame(input_data)

        # Ensure the columns match the training features
        if list(X_df.columns) != ['Latitude', 'Longitude']:
            raise ValueError("Input data must contain 'Latitude' and 'Longitude' columns.")

        # Make the prediction
        # The model is a MultiOutputRegressor, so it returns an array (N_samples, N_outputs)
        predictions_array = model.predict(X_df)

        # Define the order of the predicted output columns
        output_targets = ['PM2_5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']

        # Format the predictions into the Pydantic output schema
        results = []
        for pred in predictions_array:
            # Create a dictionary mapping the output target names to the predicted values
            prediction_dict = dict(zip(output_targets, pred.tolist()))
            results.append(PredictionOutput(**prediction_dict))

        return results

    except Exception as e:
        # Catch any modeling or data processing errors
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
