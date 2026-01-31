import pandas as pd
import json
import os

class CleaningAgent:
    def __init__(self):
        print("🧼 CleaningAgent initialized")

    def clean(self, data_path):
        print("Cleaning data...")

        df = pd.read_csv(data_path)
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

        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("reports/cleaning", exist_ok=True)

        df.to_csv("data/processed/cleaned.csv", index=False)

        with open("reports/cleaning/cleaning_report.json", "w") as f:
            json.dump(report, f, indent=4)

        return df, report


if __name__ == "__main__":
    agent = CleaningAgent()
    sample_data_path = "data/sample.csv"
    cleaned, report = agent.clean(sample_data_path)

    print("Cleaned Data:", cleaned)
    print("Report:", report)