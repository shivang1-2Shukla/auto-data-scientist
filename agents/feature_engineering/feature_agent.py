import os
import json
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class FeatureEngineeringAgent:
    def __init__(self):
        print("🧠 FeatureEngineeringAgent initialized")

    def transform(self, data_path, target_column):
        print("Starting feature engineering...")

        # load cleaned data
        df = pd.read_csv(data_path)

        # split features & target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # identify column types
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        print("Categorical columns:", categorical_cols)
        print("Numerical columns:", numerical_cols)

        # define transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )

        # fit & transform
        X_transformed = preprocessor.fit_transform(X)

        # ---- NEW: persist pipeline + metadata ----
        os.makedirs("artifacts/feature_engineering", exist_ok=True)

        joblib.dump(
            preprocessor,
            "artifacts/feature_engineering/pipeline.pkl"
        )

        metadata = {
            "categorical_columns": categorical_cols,
            "numerical_columns": numerical_cols,
            "target_column": target_column,
            "output_shape": X_transformed.shape,
        }

        with open("artifacts/feature_engineering/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print("Feature pipeline and metadata saved")

        return X_transformed, y, metadata


if __name__ == "__main__":
    agent = FeatureEngineeringAgent()

    X, y, metadata = agent.transform(
        data_path="data/processed/cleaned.csv",
        target_column="age"
    )

    print("Feature matrix shape:", X.shape)
    print("Target sample:", y.head())
    print("Metadata:", metadata)