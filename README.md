# Credit Risk Mini Model (Python + FastAPI)

## Overview
This project demonstrates a credit risk prediction system using a logistic regression model trained on synthetic loan data and deployed through a FastAPI endpoint.

The goal of this project was to simulate how financial institutions assess default risk and expose model predictions through an API for real-time usage.

---

## Key Features
- Logistic regression model for default probability prediction
- Synthetic data generation for loan risk simulation
- REST API using FastAPI for real-time predictions
- Model serialization using `joblib`
- Input validation using Pydantic schemas

---

## Technologies Used
- Python
- NumPy & Pandas
- Scikit-learn (Logistic Regression, Pipeline, Scaling)
- FastAPI
- Pydantic
- Joblib

---

## What I Learned
- How to build and train a machine learning model for classification
- How to structure a data pipeline using preprocessing + modeling
- How to evaluate model performance using ROC-AUC
- How to deploy a model via an API using FastAPI
- How backend systems serve ML predictions in real-world applications

---


### Sample Input:
```json
{
  "loan_amnt": 15000,
  "term_months": 36,
  "int_rate": 12.5,
  "fico": 700,
  "dti": 18,
  "annual_inc": 65000
}

{
  "default_probability": 0.1234
}


