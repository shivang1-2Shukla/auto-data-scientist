import pandas as pd
import json
import os
from typing import Dict


class CleaningAgent:
    def __init__(self):
        print("🧼 CleaningAgent initialized (v2)")

    def run(self, raw_data_path: str, output_path: str) -> str:
        print("🧹 Starting data cleaning...")

        df = pd.read_csv(raw_data_path)
        initial_rows = len(df)
        initial_columns = df.shape[1]

        # -----------------------------
        # 1. Remove duplicates
        # -----------------------------
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        duplicates_after = df.duplicated().sum()

        # -----------------------------
        # 2. Identify column types
        # -----------------------------
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        # -----------------------------
        # 3. Handle missing values
        # -----------------------------
        missing_before = int(df.isnull().sum().sum())

        for col in numerical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in categorical_cols:
            if df[col].isnull().any():
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])
                else:
                    df[col] = df[col].fillna("unknown")

        missing_after = int(df.isnull().sum().sum())

        # -----------------------------
        # 4. Remove constant columns
        # -----------------------------
        constant_columns = [
            col for col in df.columns if df[col].nunique(dropna=False) <= 1
        ]
        df = df.drop(columns=constant_columns)

        # -----------------------------
        # 5. Basic outlier capping (IQR)
        # -----------------------------
        outlier_capped_columns = []

        for col in numerical_cols:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
                outlier_capped_columns.append(col)

        # -----------------------------
        # 6. Final stats
        # -----------------------------
        final_rows = len(df)
        final_columns = df.shape[1]

        # -----------------------------
        # 7. Save outputs
        # -----------------------------
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs("reports/cleaning", exist_ok=True)

        df.to_csv(output_path, index=False)

        report: Dict = {
            "rows_before": initial_rows,
            "rows_after": final_rows,
            "columns_before": initial_columns,
            "columns_after": final_columns,
            "duplicates_removed": int(duplicates_before),
            "missing_values_before": missing_before,
            "missing_values_after": missing_after,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "constant_columns_removed": constant_columns,
            "outlier_capped_columns": outlier_capped_columns,
        }

        with open("reports/cleaning/cleaning_report.json", "w") as f:
            json.dump(report, f, indent=4)

        print("✅ Cleaning completed")
        return output_path
