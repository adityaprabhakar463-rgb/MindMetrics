# MindMetrics

A machine learning web API that predicts student productivity scores 
and burnout risk based on daily study habits.

## What it does

Input your daily study data and get back:
- A **productivity score** (0–100) based on your habits
- A **burnout risk assessment** with probability score

## How it works

Two separate ML models trained on student study session data:

**Productivity model** — Linear Regression trained on:
- Study hours, sleep hours, mood rating
- Distraction level, task difficulty, exam proximity

**Burnout model** — Classification model with a custom probability 
threshold, predicting whether current habits indicate burnout risk.

Both models are served via a Flask REST API with two endpoints:
```
POST /api/predict/productivity  → returns productivity score + interpretation
POST /api/predict/burnout       → returns burnout risk + probability
```

## Tech stack

- Python — data processing and model training
- Scikit-learn — Linear Regression + classification models
- Pandas + NumPy — data manipulation
- Flask — REST API server
- Flask-CORS — cross-origin support for frontend integration
- Joblib — model serialization

## Project structure
```
MindMetrics/
├── app.py                      — Flask API server
├── data/
│   └── study_data.csv          — training dataset
├── models/
│   ├── productivity_model.pkl  — trained regression model
│   └── burnout_model.pkl       — trained classification model
└── src/
    ├── data_processing.py      — data cleaning pipeline
    ├── train_model.py          — productivity model training
    ├── train_burnout_model.py  — burnout model training
    └── eda.py                  — exploratory data analysis
```

## Running locally
```bash
# install dependencies
pip install -r requirements.txt

# train the models (first time only)
python src/train_model.py
python src/train_burnout_model.py

# start the API server
python app.py
# server runs at http://localhost:5001
```

## Example request
```bash
curl -X POST http://localhost:5001/api/predict/productivity \
  -H "Content-Type: application/json" \
  -d '{
    "Study Hours": 6.5,
    "Sleep Hours": 7.0,
    "Mood": 4,
    "Distraction": 2,
    "Difficulty": 3,
    "Exam Proximity": 15
  }'
```

## Note

This was an exploratory project built to learn full stack ML concepts 
— Flask API design, model serialization, and REST endpoint structure. 
A proper rebuild with a frontend UI, better dataset, and model 
evaluation is planned for a future version.
