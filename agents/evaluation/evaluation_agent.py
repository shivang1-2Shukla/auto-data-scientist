import os
import json
import joblib
import numpy as np

from sklearn.model_selection import cross_val_score


class EvaluationAgent:
    def __init__(self, task_type="regression"):
        print(f"📊 EvaluationAgent initialized ({task_type})")
        self.task_type = task_type

    def run(self, X, y, cv=5):
        print("Running cross-validation...")

        model = joblib.load("artifacts/model/model.pkl")

        n_samples = X.shape[0]
        effective_cv = min(cv, n_samples)

        if effective_cv < 2:
            raise ValueError(
                f"Not enough samples for evaluation: n_samples={n_samples}"
            )

        print(f"Using cv={effective_cv} for evaluation")

        if self.task_type == "regression":
            scores = cross_val_score(
                model,
                X,
                y,
                cv=effective_cv,
                scoring="neg_root_mean_squared_error"
            )

            metric_scores = -scores
            metric_name = "rmse"
        elif self.task_type == "classification":
            scores = cross_val_score(
                model,
                X,
                y,
                cv=effective_cv,
                scoring="accuracy"
            )
            metric_scores = scores
            metric_name = "accuracy"
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        report = {
            "metric": metric_name,
            "cv_folds": effective_cv,
            "n_samples": n_samples,
            f"mean_{metric_name}": float(np.mean(metric_scores)),
            f"std_{metric_name}": float(np.std(metric_scores)),
            "all_scores": metric_scores.tolist()
        }

        os.makedirs("artifacts/evaluation", exist_ok=True)

        with open("artifacts/evaluation/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=4)

        print("✅ Evaluation completed")
        return report