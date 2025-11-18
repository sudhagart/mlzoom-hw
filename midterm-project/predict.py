#!/usr/bin/env python
# coding: utf-8

# ## Diabetes Prediction DataSet Analysis

# ### Dataset
# 
# URL: https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset
# 
# ### Attributes 
# - columnname uniquevalues numnulls type
# - year 7 0 int64
# - gender 3 0 object
# - age 102 0 float64
# - location 55 0 object
# - race_africanamerican 2 0 int64
# - race_asian 2 0 int64
# - race_caucasian 2 0 int64
# - race_hispanic 2 0 int64
# - race_other 2 0 int64
# - hypertension 2 0 int64
# - heart_disease 2 0 int64
# - smoking_history 6 0 object
# - bmi 4247 0 float64
# - hba1c_level 18 0 float64
# - blood_glucose_level 18 0 int64
# - diabetes 2 0 int64
# 

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

import pandas as pd
import pickle
import uvicorn
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

class Patient(BaseModel):
    year: int = Field(..., ge=1900, le=2100)
    age: float = Field(..., ge=0.0)
    location: str
    gender: Literal["male", "female"]
    race_africanamerican: Literal[0, 1]
    race_asian: Literal[0, 1]
    race_caucasian: Literal[0, 1]
    race_hispanic: Literal[0, 1]
    race_other: Literal[0, 1]
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    smoking_history: Literal["Never", "Former", "Current"]
    bmi: float = Field(..., ge=0.0)
    hba1c_level: float = Field(..., ge=0.0)
    blood_glucose_level: int = Field(..., ge=0)


class PredictResponse(BaseModel):
    diabetes_probability: float
    diabetes_prediction: bool

targetcolumn = 'diabetes'
modelfile = 'diabetes_model.bin'

app = FastAPI(title="Diabetes-prediction")

def load_model(modelfile):
    with open(modelfile, 'rb') as f_in:
        (dv, model), threshold_final = pickle.load(f_in)
    return dv, model, threshold_final

dv, model, threshold_final = load_model(modelfile)

print ("Model and threshold loaded : ", threshold_final)
def predict_diabetes(patient_data: dict) -> dict:
    df = pd.DataFrame([patient_data])
    print (df)
 
    X_dict = dv.transform(df.to_dict(orient='records')) 
    dmatrix = xgb.DMatrix(X_dict, feature_names=dv.get_feature_names_out().tolist())       
    xgb_pred = model.predict(dmatrix)
    probability = float(xgb_pred[0])
    prediction = int(probability >= threshold_final)
    print (probability, prediction)
    return PredictResponse(
        diabetes_probability=probability,
        diabetes_prediction=prediction
    )

#sample input for testing
sample_input = {
    "year": 2020,
    "gender": "Female",
    "age": 45.0,
    "location": "Washington",
    "race_africanamerican": 0,
    "race_asian": 0,
    "race_caucasian": 1,
    "race_hispanic": 0,
    "race_other": 0,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "Never",
    "bmi": 28.5,
    "hba1c_level": 6.7,
    "blood_glucose_level": 210
}
#print(predict_diabetes(sample_input))


@app.post("/predict")
def predict(client: Patient) -> PredictResponse:
    prediction = predict_diabetes(client.dict())

    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)