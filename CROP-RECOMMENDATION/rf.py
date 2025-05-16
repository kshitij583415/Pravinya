# Import necessary libraries
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI()

# Load data and models if pre-trained, or initialize
X_train, X_test, y_train, y_test = ... # Replace with actual data loading
le = LabelEncoder()
le.fit(y_train)
RF_model = RandomForestClassifier(n_estimators=20, random_state=5)
RF_model.fit(X_train, y_train)

# Input data model for prediction
class CropPredictionInput(BaseModel):
    features: List[float]

@app.post("/train")
async def train_model():
    """Endpoint to retrain the RandomForest model with current data"""
    RF_model.fit(X_train, y_train)
    return {"message": "Model retrained successfully"}

@app.post("/predict")
async def predict_crop(data: CropPredictionInput):
    """Predict the crop based on provided features"""
    features = np.array(data.features).reshape(1, -1)
    label = RF_model.predict(features)[0]
    crop_name = le.inverse_transform([label])[0]
    return {"predicted_crop": crop_name}

@app.get("/accuracy")
async def get_accuracy():
    """Get accuracy per crop type"""
    predicted_labels = RF_model.predict(X_test)
    accuracy_per_crop = []
    crop_labels = le.inverse_transform(sorted(np.unique(y_test)))
    for crop_label in crop_labels:
        indices = (y_test == le.transform([crop_label])[0])
        accuracy = accuracy_score(y_test[indices], predicted_labels[indices])
        accuracy_per_crop.append({"crop": crop_label, "accuracy": accuracy})
    return {"accuracy_per_crop": accuracy_per_crop}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
