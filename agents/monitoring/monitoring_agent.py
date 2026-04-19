import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class MonitoringAgent:
    def __init__(self):
        print("🕵️ MonitoringAgent initialized")
        self.reports_dir = "reports"

    def generate_drift_report(self, reference_data_path, current_data_path, output_filename="data_drift_report.html"):
        print("Generating data drift report...")
        
        if not os.path.exists(reference_data_path) or not os.path.exists(current_data_path):
            print("Warning: Missing data files for monitoring. Skipping report generation.")
            return None

        # Load datasets
        reference_data = pd.read_csv(reference_data_path)
        current_data = pd.read_csv(current_data_path)

        # Create report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Save report
        os.makedirs(self.reports_dir, exist_ok=True)
        report_path = os.path.join(self.reports_dir, output_filename)
        report.save_html(report_path)
        
        print(f"✅ Data drift report generated at {report_path}")
        return report_path

if __name__ == "__main__":
    agent = MonitoringAgent()
    # Example usage (requires existing data):
    # agent.generate_drift_report("data/processed/cleaned.csv", "data/processed/cleaned.csv")
