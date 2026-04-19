from agents.feature_engineering.feature_agent import FeatureEngineeringAgent
from agents.cleaning.cleaning_agent import CleaningAgent
from agents.automl.automl_agent import AutoMLAgent
from agents.evaluation.evaluation_agent import EvaluationAgent
from agents.deployment.deployment_agent import DeploymentAgent
from agents.monitoring.monitoring_agent import MonitoringAgent


class Orchestrator:
    def __init__(self, task_type="regression"):
        print("🧩 Orchestrator initialized")
        self.task_type = task_type

        self.cleaning_agent = CleaningAgent()
        self.feature_agent = FeatureEngineeringAgent(task_type=task_type)
        self.automl_agent = AutoMLAgent(task_type=task_type)
        self.evaluation_agent = EvaluationAgent(task_type=task_type)
        self.deployment_agent = DeploymentAgent()
        self.monitoring_agent = MonitoringAgent()

    def run_training_pipeline(self, raw_data_path, target_column):
        print("🚀 Starting full training pipeline")

        # Step 1: Cleaning
        cleaned_data_path = self.cleaning_agent.run(
            raw_data_path=raw_data_path,
            output_path="data/processed/cleaned.csv"
        )

        # Step 2: Feature Engineering
        X, y, fe_metadata = self.feature_agent.transform(
            data_path=cleaned_data_path,
            target_column=target_column
        )

        # Step 3: AutoML
        model, automl_report = self.automl_agent.run(X, y)

        # Step 4: Evaluation
        eval_report = self.evaluation_agent.run(X, y)

        # Step 5: Deployment
        deployment_dir = self.deployment_agent.deploy()
        
        # Step 6: Monitoring (Initial profile generation)
        drift_report_path = self.monitoring_agent.generate_drift_report(
            reference_data_path=cleaned_data_path,
            current_data_path=cleaned_data_path  # Simulating new data for demonstration
        )

        print("✅ Training pipeline completed")

        return {
            "cleaned_data_path": cleaned_data_path,
            "feature_metadata": fe_metadata,
            "automl_report": automl_report,
            "evaluation_report": eval_report,
            "deployment_dir": deployment_dir,
            "drift_report_path": drift_report_path
        }
