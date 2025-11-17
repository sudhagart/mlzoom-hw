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

# ### Download and Inspect Data

#!/bin/bash


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import pickle
import os

targetcolumn = 'diabetes'
C_final = 0.81
threshold_final = 0.45
modelfile = 'diabetes_model.bin'

def load_data():
    os.system('wget  https://www.kaggle.com/api/v1/datasets/download/priyamchoksi/100000-diabetes-clinical-dataset -O diabetes_data.zip')
    os.system('rm diabetes_dataset.csv')
    os.system('unzip diabetes_data.zip -d .')
    os.system('rm diabetes_data.zip')

    data = pd.read_csv('diabetes_dataset.csv')
    data.columns = data.columns.str.replace(':', '_').str.lower()
    return data

data = load_data()

def train_model(df):
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical.remove(targetcolumn)
    categorical = df.select_dtypes(include=['object']).columns.tolist()

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_full_train = df_full_train.reset_index(drop=True)
    y_train = df_train[targetcolumn].values
    y_val = df_val[targetcolumn].values
    y_test = df_test[targetcolumn].values
    y_full_train = df_full_train[targetcolumn].values
    del df_train[targetcolumn]
    del df_val[targetcolumn]
    del df_test[targetcolumn]

    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear', C=C_final, max_iter=1000, random_state=1)
    )
    X_full_train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
    pipeline.fit(X_full_train_dict, y_full_train)
    return pipeline

def save_model(pipeline, modelfile):
    with open(modelfile, 'wb') as f_out:
        pickle.dump((pipeline, threshold_final), f_out)

data = load_data()
model = train_model(data)
save_model(model, modelfile)