
# Diabetes Prediction Model 

Use the Kaggle dataset to train various diabetes prediction models. 

This can be used to predict what attributes of a patient can lead to Diabetes. 

### Dataset

URL: https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset

### Attributes 
- columnname uniquevalues numnulls type
- year 7 0 int64
- gender 3 0 object
- age 102 0 float64
- location 55 0 object
- race_africanamerican 2 0 int64
- race_asian 2 0 int64
- race_caucasian 2 0 int64
- race_hispanic 2 0 int64
- race_other 2 0 int64
- hypertension 2 0 int64
- heart_disease 2 0 int64
- smoking_history 6 0 object
- bmi 4247 0 float64
- hba1c_level 18 0 float64
- blood_glucose_level 18 0 int64
- diabetes 2 0 int64

### Analysis Results from various models
- Linear Regression ROC AUC: 0.804, RMSE: 0.207
- Decision Tree --- ROC AUC: 0.967, RMSE: 0.161
- Random Forest --- ROC AUC: 0.969, RMSE: 0.161
- XGBoost Model --- ROC AUC: 0.973, RMSE: 0.158 

### Requirements
- Internet access to download the dataset from Kaggle

### Training the model
- Checkout the repo
- cd midterm-project
- uv sync 
- uv run python train.py 

### Containerizing the model in docker and running it
- After the model is trained
- docker build -t diabetes-prediction .
- docker run -it --rm -p 9696:9696 diabetes-prediction

### FLY Hosting Live
- Go to https://quiet-star-7278.fly.dev/docs
- Try out the Predict function with the following input. 

### Deploy to your own FLY hosting
- cd midterm-project
- install fly and login 
- - Reference here - https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/workshop/README.md#deployment
- fly launch --generate-name
- go to deployed url/docs

### Sample Input
{
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