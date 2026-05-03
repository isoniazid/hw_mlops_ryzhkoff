from fastapi import FastAPI
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.get("/health")
def health():
    return {"status": "ok", "version": "v1.0.0"}

@app.post("/predict")
def predict(data: dict):
    x = np.array(data["x"]).reshape(1, -1)
    pred = model.predict(x)
    return {"prediction": int(pred[0])}