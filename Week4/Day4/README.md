# Week 4 Day 4

## Topic

FastAPI Model Serving Endpoint

## Objective

Deploy a trained Logistic Regression model as a REST API using FastAPI.

## Files

- train_model.py
- app.py
- logistic_model.joblib
- requirements.txt

## Technologies

- Python
- FastAPI
- Scikit-learn
- Joblib

## Run

Train the model:

```bash
python train_model.py
```

Start the API:

```bash
uvicorn app:app --reload
```

API documentation:

```
http://127.0.0.1:8000/docs
```