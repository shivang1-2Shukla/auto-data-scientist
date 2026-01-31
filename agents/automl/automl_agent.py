import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class AutoMLAgent:
    def __init__(self, task_type="regression"):
        print("🤖 AutoMLAgent initialized")
        self.task_type = task_type

        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        }

    def run(self, X, y):
        print("Training models...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            rmse = root_mean_squared_error(y_val, preds)

            results[name] = {"rmse": rmse}

        best_model_name = min(results, key=lambda x: results[x]["rmse"])
        best_model = self.models[best_model_name]

        os.makedirs("artifacts/model", exist_ok=True)
        joblib.dump(best_model, "artifacts/model/model.pkl")

        report = {
            "task_type": self.task_type,
            "metric": "rmse",
            "results": results,
            "best_model": best_model_name
        }

        with open("artifacts/model/training_report.json", "w") as f:
            json.dump(report, f, indent=4)

        print("✅ AutoML completed. Best model:", best_model_name)
        return best_model, report