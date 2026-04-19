import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, accuracy_score

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier


class AutoMLAgent:
    def __init__(self, task_type="regression"):
        print(f"🤖 AutoMLAgent initialized ({task_type})")
        self.task_type = task_type

        if self.task_type == "regression":
            self.metric_name = "rmse"
            self.models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.01),
                "ElasticNet": ElasticNet(),
                "RandomForest": RandomForestRegressor(),
                "ExtraTrees": ExtraTreesRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "KNN": KNeighborsRegressor(),
                "MLP": MLPRegressor(max_iter=500)
            }
        elif self.task_type == "classification":
            self.metric_name = "accuracy"
            self.models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(),
                "ExtraTrees": ExtraTreesClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "KNN": KNeighborsClassifier(),
                "MLP": MLPClassifier(max_iter=500)
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def run(self, X, y):
        print("Training models...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            if self.task_type == "regression":
                score = root_mean_squared_error(y_val, preds)
            else:
                score = accuracy_score(y_val, preds)

            results[name] = {self.metric_name: score}

        if self.task_type == "regression":
            best_model_name = min(results, key=lambda x: results[x][self.metric_name])
        else:
            best_model_name = max(results, key=lambda x: results[x][self.metric_name])

        best_model = self.models[best_model_name]

        os.makedirs("artifacts/model", exist_ok=True)
        joblib.dump(best_model, "artifacts/model/model.pkl")

        report = {
            "task_type": self.task_type,
            "metric": self.metric_name,
            "results": results,
            "best_model": best_model_name
        }

        with open("artifacts/model/training_report.json", "w") as f:
            json.dump(report, f, indent=4)

        print("✅ AutoML completed. Best model:", best_model_name)
        return best_model, report