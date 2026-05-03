import json
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

#### ИМИТИРУЕМ СОХРАНЕНИЕ АРТЕФАКТОВ

iris = load_iris()
X, y = iris.data, iris.target

params = {"n_estimators": 100, "random_state": 42}

model = RandomForestClassifier(**params)
model.fit(X, y)

joblib.dump(model, "artifacts/model.joblib")

#
with open("artifacts/params.json", "w") as f:
    json.dump(params, f)

print("Model and params saved")