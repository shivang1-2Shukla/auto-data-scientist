import os
import json
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


class FeatureEngineeringAgent:
    def __init__(self, task_type="regression"):
        print(f"🧠 FeatureEngineeringAgent initialized ({task_type})")
        self.task_type = task_type

    def transform(self, data_path, target_column):
        print("Starting feature engineering...")

        # load cleaned data
        df = pd.read_csv(data_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # Check for regression on strings
        if self.task_type == "regression" and df[target_column].dtype == 'object':
            raise ValueError(
                f"Error: You selected Regression, but the target column '{target_column}' contains text data. "
                "Regression algorithms only work on numbers. Please change the Task Type to Classification, or choose a numeric target column."
            )

        # split features & target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Process Target (y) if Classification
        label_encoder = None
        encoder_path = "artifacts/feature_engineering/target_encoder.pkl"
        os.makedirs("artifacts/feature_engineering", exist_ok=True)
        
        if self.task_type == "classification":
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), name=target_column)

            # Save the LabelEncoder so we can inverse_transform later
            joblib.dump(label_encoder, encoder_path)
            print("Target column label-encoded for classification.")
        else:
            # Clean up old encoder if it exists from a previous classification run
            if os.path.exists(encoder_path):
                os.remove(encoder_path)

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
