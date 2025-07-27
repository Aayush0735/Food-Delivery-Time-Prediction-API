# --- app.py ---

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the FastAPI app
app = FastAPI()

# 2. Load the trained model and columns
model = joblib.load('delivery_time_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 3. Define the input data structure using Pydantic
class DeliveryData(BaseModel):
    Delivery_person_Age: int
    Delivery_person_Ratings: float
    distance_km: float
    Type_of_order: str # e.g., "Snack", "Drinks", "Buffet", "Meal"
    Type_of_vehicle: str # e.g., "motorcycle", "scooter", "electric_scooter"

# 4. Create the /predict endpoint
@app.post('/predict')
def predict_delivery_time(data: DeliveryData):
    # Convert incoming data into a pandas DataFrame
    input_data = pd.DataFrame([data.dict()])

    # One-hot encode the categorical features
    # This must match the encoding done during training
    input_data = pd.get_dummies(input_data)

    # Reindex the DataFrame to match the model's expected columns
    # This adds missing columns (with a value of 0) and ensures the correct order
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the prediction
    return {'predicted_delivery_time_min': round(prediction[0], 2)}