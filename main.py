from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib

import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import itertools
import uvicorn

from config import MODEL_PATH, MODEL_ENCODER, MODEL_SCALER, MODEL_NAME

app = FastAPI()



# Load encoder and scaler
enc = joblib.load(MODEL_PATH + MODEL_ENCODER)
scaler = joblib.load( MODEL_PATH + MODEL_SCALER)

model = tf.keras.models.load_model(MODEL_PATH + MODEL_NAME)

class HotelPredictRequest(BaseModel):
  office_location: str
  event_months: list
  group_size: int
  days_durations: list
  total_budget: int
  top_n: int

@app.get('/ping')
async def ping():
    return {"message": "ok"}


@app.post("/predict")
async def predict(request: HotelPredictRequest):
    if not request.event_months or not request.days_durations:
        raise HTTPException(status_code=400, detail="Event months and days durations must be provided and non-empty.")

    # Generate prefixed hotel IDs (example: 'Paris_1' to 'Paris_5', etc.)
    cities = ['Paris', 'Cancun', 'Niagara']
    numbers = range(1, 6)
    prefixed_hotel_ids = [f"{city}_{number}" for city, number in itertools.product(cities, numbers)]

    top_hotels = predict_top_hotels(
        request.office_location,
        request.event_months,
        request.group_size,
        request.days_durations,
        request.total_budget,
        prefixed_hotel_ids,
        request.top_n
    )

    return {"top_hotels": top_hotels.to_dict(orient='records')}


def predict_top_hotels(office_location, event_months, group_size, days_durations, total_budget, prefixed_hotel_ids, top_n=3):
  combinations = list(itertools.product(event_months, days_durations, prefixed_hotel_ids))
  new_data = pd.DataFrame(combinations, columns=['event_month', 'days_duration', 'prefixed_hotel_id'])
  new_data['office_location'] = office_location
  new_data['group_size'] = group_size
  new_data['total_budget'] = total_budget

  # Transform categorical and numerical data using pre-trained encoder and scaler
  categorical_data_new = enc.transform(new_data[['office_location', 'event_month', 'prefixed_hotel_id']])
  numerical_data_new = scaler.transform(new_data[['group_size', 'days_duration', 'total_budget']])
  X_new = np.hstack([categorical_data_new.toarray(), numerical_data_new])

    # Make predictions
  predictions = model.predict(X_new)

    # Attach predictions to the DataFrame
  new_data['predicted_nps'] = predictions[:, 0]  # Assuming the prediction output is the first column

  return new_data.sort_values(by='predicted_nps', ascending=False).head(top_n)


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8080)