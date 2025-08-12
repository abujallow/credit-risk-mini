

Tiny demo: train a logistic regression on synthetic loan data and serve predictions with FastAPI.


python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload

