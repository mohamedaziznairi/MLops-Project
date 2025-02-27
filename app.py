from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List  # Import List for compatibility with Python 3.8
import pandas as pd  # Import pandas

# Load the trained model
MODEL_PATH = "best_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI()

# Request model for predictions
class PredictionInput(BaseModel):
    features: List[float]  # Input features for prediction, must be a list of floats
# Request model for retraining
class RetrainInput(BaseModel):
    new_data: List[dict]  # List of dictionaries (rows of new training data)
# Prediction route
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Prepare the features for prediction (reshape for model input)
        features = np.array(data.features).reshape(1, -1)
        
        # Predict with the model
        prediction = model.predict(features)
        
        # Return the prediction in JSON-compatible format
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        # Handle any errors that occur during prediction
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
@app.post("/retrain")
def retrain(data: RetrainInput):
    try:
        df = pd.DataFrame(data.new_data)
        
        # Debug: Print first rows of the new dataset
        print("Received data for retraining:\n", df.head())

        # Check if 'target' column exists
        if "target" not in df.columns:
            raise ValueError("Missing 'target' column in the provided data.")

        X, y = df.drop("target", axis=1), df["target"]

        print("Features shape:", X.shape)
        print("Target shape:", y.shape)

        global model
        model.fit(X, y)  # Re-train the model
        joblib.dump(model, MODEL_PATH)  # Save updated model

        return {"message": "Model retrained successfully!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Retraining error: {e}")
@app.websocket("/predict_ws")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive data from the client (expecting features as a list of floats)
            data = await websocket.receive_json()
            features = np.array(data["features"]).reshape(1, -1)
            
            # Predict with the model
            prediction = model.predict(features)
            
            # Send prediction back to the client
            await websocket.send_json({"prediction": prediction.tolist()})
    
    except WebSocketDisconnect:
        print("Client disconnected")