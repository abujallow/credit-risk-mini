from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="Credit Risk Mini-Model")

class Features(BaseModel):
    loan_amnt: float = Field(..., ge=0)
    term_months: int
    int_rate: float
    fico: float
    dti: float
    annual_inc: float = Field(..., ge=0)

model = joblib.load("artifacts/model.pkl")

@app.get("/")
def root():
    return {"ok": True, "message": "Use POST /predict"}

@app.post("/predict")
def predict(f: Features):
    X = [[f.loan_amnt, f.term_months, f.int_rate, f.fico, f.dti, f.annual_inc]]
    prob = float(model.predict_proba(X)[0, 1])
    return {"default_probability": round(prob, 4)}
