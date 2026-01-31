from agents.feature_engineering.feature_agent import FeatureEngineeringAgent
from agents.automl.automl_agent import AutoMLAgent
from agents.evaluation.evaluation_agent import EvaluationAgent


class Orchestrator:
    def __init__(self, task_type="regression"):
        print("🧩 Orchestrator initialized")
        self.task_type = task_type

        self.feature_agent = FeatureEngineeringAgent()
        self.automl_agent = AutoMLAgent(task_type=task_type)
        self.evaluation_agent = EvaluationAgent(task_type=task_type)

    def run_training_pipeline(self, data_path, target_column):
        print("🚀 Starting training pipeline")

        # Step 1: Feature Engineering
        X, y, fe_metadata = self.feature_agent.transform(
            data_path=data_path,
            target_column=target_column
        )

        # Step 2: AutoML
        model, automl_report = self.automl_agent.run(X, y)

        # Step 3: Evaluation
        eval_report = self.evaluation_agent.run(X, y)

        print("✅ Training pipeline completed")

        return {
            "feature_metadata": fe_metadata,
            "automl_report": automl_report,
            "evaluation_report": eval_report
        }