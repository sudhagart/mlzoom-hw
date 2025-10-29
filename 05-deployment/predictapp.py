import pickle
from fastapi import FastAPI 
import uvicorn


from sklearn.pipeline import Pipeline

app = FastAPI(title="Predict App")

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

def predict_single(record: dict) -> float:
    prediction = pipeline.predict_proba([record])
    return prediction[0, 1].round(3)

@app.post("/predict")
def predict(record: dict):
    prediction = predict_single(record)
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)