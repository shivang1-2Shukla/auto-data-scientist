from agents.feature_engineering.feature_agent import FeatureEngineeringAgent
from agents.cleaning.cleaning_agent import CleaningAgent
from agents.automl.automl_agent import AutoMLAgent
from agents.evaluation.evaluation_agent import EvaluationAgent


class Orchestrator:
    def __init__(self, task_type="regression"):
        print("🧩 Orchestrator initialized")
        self.task_type = task_type

        self.cleaning_agent = CleaningAgent()
        self.feature_agent = FeatureEngineeringAgent()
        self.automl_agent = AutoMLAgent(task_type=task_type)
        self.evaluation_agent = EvaluationAgent(task_type=task_type)

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

        print("✅ Training pipeline completed")

        return {
            "cleaned_data_path": cleaned_data_path,
            "feature_metadata": fe_metadata,
            "automl_report": automl_report,
            "evaluation_report": eval_report
        }
