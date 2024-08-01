from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn

# Inisialisasi server FastAPI
app = FastAPI()

# Definisikan input data model
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

# Load model dari pickle file
model_path = 'iris_model.pkl'
model = joblib.load(model_path)

# Endpoint untuk prediksi
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input data to numpy array
        input_array = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        # Buat prediksi
        prediction = model.predict(input_array)
        predicted_class = int(prediction[0])
        if (predicted_class == 0):
            return {"class" : "Iris-setosa"}
        elif (predicted_class == 1):
            return {"class" : "Iris-versicolor"}
        elif (predicted_class == 2):
            return {"class" : "Iris-virginica"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug")

