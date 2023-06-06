from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Define the request payload schema
class PredictionRequest(BaseModel):
    input_data: list

# Load the trained machine learning model
model = joblib.load('trained_model.pkl')

# Create a FastAPI instance
app = FastAPI()

# Endpoint for making predictions
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = request.input_data
        # Perform data validation if needed
        if not isinstance(input_data, list):
            raise HTTPException(status_code=400, detail="Invalid input data. Expected a list.")
        
        # Perform prediction using the loaded model
        prediction = model.predict([input_data])[0]
        
        # Return the prediction as the response
        return {"prediction": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
