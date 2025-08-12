import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from pathlib import Path
import joblib

rng = np.random.default_rng(7)
n = 5000

# Synthetic features
loan_amnt = rng.normal(12000, 4000, n).clip(1000, 40000)
term_months = rng.choice([36, 60], n, p=[0.7, 0.3])
int_rate = rng.normal(13, 4, n).clip(5, 30)
fico = rng.normal(690, 50, n).clip(540, 850)
dti = rng.normal(18, 8, n).clip(0, 45)
annual_inc = rng.normal(60000, 25000, n).clip(10000, 250000)

# Latent probability of default
logit = (
    0.00008 * loan_amnt
    + 0.015 * (term_months == 60).astype(float)
    + 0.10 * (int_rate - 10)
    - 0.01 * (fico - 680)
    + 0.02 * (dti - 18)
    - 0.000002 * (annual_inc - 60000)
)
p_default = 1 / (1 + np.exp(-logit))
default = rng.binomial(1, p_default)

df = pd.DataFrame({
    "loan_amnt": loan_amnt,
    "term_months": term_months,
    "int_rate": int_rate,
    "fico": fico,
    "dti": dti,
    "annual_inc": annual_inc,
    "default": default,
})

X = df.drop(columns=["default"])
y = df["default"]

pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("lr", LogisticRegression(max_iter=1000)),
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_tr, y_tr)

pred = pipe.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, pred)

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(pipe, "artifacts/model.pkl")
Path("artifacts/metrics.txt").write_text(f"AUC: {auc:.3f}\n", encoding="utf-8")

print(f"Saved model to artifacts/model.pkl — Test AUC: {auc:.3f}")
