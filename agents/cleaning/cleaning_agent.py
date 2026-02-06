import pandas as pd
import json
import os


class CleaningAgent:
    def __init__(self):
        print("🧼 CleaningAgent initialized")

    def run(self, raw_data_path, output_path):
        print("Cleaning data...")

        df = pd.read_csv(raw_data_path)
        print(f"Rows before cleaning: {len(df)}")

        # remove duplicates
        df = df.drop_duplicates()

        # handle missing values
        missing_before = df.isnull().sum().sum()
        df = df.ffill()
        missing_after = df.isnull().sum().sum()

        print(f"Missing values before: {missing_before}")
        print(f"Missing values after: {missing_after}")

        report = {
            "duplicates_removed": "handled",
            "missing_values_before": int(missing_before),
            "missing_values_after": int(missing_after)
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs("reports/cleaning", exist_ok=True)

        df.to_csv(output_path, index=False)

        with open("reports/cleaning/cleaning_report.json", "w") as f:
            json.dump(report, f, indent=4)

        return output_path
