import pandas as pd
import json
import os
from typing import Dict, List


class CleaningAgent:
    def __init__(
        self,
        max_missing_ratio: float = 0.4,
        max_unique_ratio: float = 0.95,
        outlier_iqr_multiplier: float = 1.5,
    ):
        print("🧼 CleaningAgent initialized (v3 – production-grade)")

        self.max_missing_ratio = max_missing_ratio
        self.max_unique_ratio = max_unique_ratio
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

    # -----------------------------
    # Validation (fail fast)
    # -----------------------------
    def _validate_dataframe(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("❌ Dataset is empty")

        if df.shape[1] < 2:
            raise ValueError("❌ Dataset must contain at least 2 columns")

        if df.isnull().all().any():
            raise ValueError("❌ One or more columns are fully null")

    # -----------------------------
    # Main entry
    # -----------------------------
    def run(self, raw_data_path: str, output_path: str) -> str:
        print("🧹 Starting advanced data cleaning...")

        df = pd.read_csv(raw_data_path)

        self._validate_dataframe(df)

        report: Dict = {}
        report["rows_before"] = len(df)
        report["columns_before"] = df.shape[1]

        # -----------------------------
        # 1. Remove duplicates
        # -----------------------------
        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates()
        report["duplicates_removed"] = duplicates_removed

        # -----------------------------
        # 2. Infer column types
        # -----------------------------
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        report["numerical_columns"] = numerical_cols
        report["categorical_columns"] = categorical_cols

        # -----------------------------
        # 3. Drop high-missing columns
        # -----------------------------
        high_missing_cols: List[str] = []
        for col in df.columns:
            if df[col].isnull().mean() > self.max_missing_ratio:
                high_missing_cols.append(col)

        df = df.drop(columns=high_missing_cols)
        report["high_missing_columns_dropped"] = high_missing_cols

        # Update column lists
        numerical_cols = [c for c in numerical_cols if c in df.columns]
        categorical_cols = [c for c in categorical_cols if c in df.columns]

        # -----------------------------
        # 4. Handle missing values (column-aware)
        # -----------------------------
        missing_before = int(df.isnull().sum().sum())

        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())

        for col in categorical_cols:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "unknown")

        missing_after = int(df.isnull().sum().sum())
        report["missing_values_before"] = missing_before
        report["missing_values_after"] = missing_after

        # -----------------------------
        # 5. Remove constant columns
        # -----------------------------
        constant_columns = [
            col for col in df.columns if df[col].nunique(dropna=False) <= 1
        ]
        df = df.drop(columns=constant_columns)
        report["constant_columns_removed"] = constant_columns

        # -----------------------------
        # 6. Detect & drop ID-like columns
        # -----------------------------
        id_like_columns = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > self.max_unique_ratio:
                id_like_columns.append(col)

        df = df.drop(columns=id_like_columns)
        report["id_like_columns_removed"] = id_like_columns

        # Update numeric columns again
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # -----------------------------
        # 7. Outlier capping (IQR)
        # -----------------------------
        outlier_summary = {}

        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR <= 0:
                continue

            lower = Q1 - self.outlier_iqr_multiplier * IQR
            upper = Q3 + self.outlier_iqr_multiplier * IQR

            outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            df[col] = df[col].clip(lower, upper)

            outlier_summary[col] = outliers

        report["outliers_capped"] = outlier_summary

        # -----------------------------
        # 8. Normalize categorical strings
        # -----------------------------
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

        # -----------------------------
        # 9. Final stats
        # -----------------------------
        report["rows_after"] = len(df)
        report["columns_after"] = df.shape[1]
        report["final_columns"] = list(df.columns)

        # -----------------------------
        # 10. Persist outputs
        # -----------------------------
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs("reports/cleaning", exist_ok=True)

        df.to_csv(output_path, index=False)

        with open("reports/cleaning/cleaning_report.json", "w") as f:
            json.dump(report, f, indent=4)

        print("✅ Advanced cleaning completed successfully")
        return output_path
