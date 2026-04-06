# agents/monitoring_agent.py

import json
import os
from datetime import datetime


class MonitoringAgent:
    def __init__(self, report_dir="reports"):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_data = {}

    def start_run(self):
        self.run_data["run_id"] = self.run_id
        self.run_data["start_time"] = datetime.now().isoformat()
        self.run_data["status"] = "RUNNING"

    def log_metrics(self, metrics: dict):
        self.run_data["metrics"] = metrics

    def log_artifacts(self, artifacts: dict):
        self.run_data["artifacts"] = artifacts

    def end_run(self, status="SUCCESS"):
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data["status"] = status

        file_path = os.path.join(self.report_dir, f"run_{self.run_id}.json")
        with open(file_path, "w") as f:
            json.dump(self.run_data, f, indent=4)