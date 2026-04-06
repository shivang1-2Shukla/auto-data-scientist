from agents.feature_engineering.feature_agent import FeatureEngineeringAgent
from agents.cleaning.cleaning_agent import CleaningAgent
from agents.automl.automl_agent import AutoMLAgent
from agents.evaluation.evaluation_agent import EvaluationAgent
from agents.monitoring.monitoring_agent import MonitoringAgent

class Orchestrator:
    def __init__(self, task_type="regression"):
        print("🧩 Orchestrator initialized")
        self.task_type = task_type

        self.cleaning_agent = CleaningAgent()
        self.feature_agent = FeatureEngineeringAgent()
        self.automl_agent = AutoMLAgent(task_type=task_type)
        self.evaluation_agent = EvaluationAgent(task_type=task_type)
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
        
        # Optional: log dataset shape
        self.monitoring_agent.log_metrics({
            "num_rows": X.shape[0],
            "num_features": X.shape[1]
        })

        
        # Step 3: AutoML
        model, automl_report = self.automl_agent.run(X, y)

        # Step 4: Evaluation
        eval_report = self.evaluation_agent.run(X, y)

         # ✅ Aggregate metrics
        self.monitoring_agent.log_metrics({
            "automl": automl_report,
            "evaluation": eval_report
        })

        # ✅ Log artifacts (adjust paths if needed)
        self.monitoring_agent.log_artifacts({
            "cleaned_data": cleaned_data_path,
            "feature_metadata": "artifacts/metadata.json",
            "pipeline": "artifacts/pipeline.pkl",
            "model": "models/best_model.pkl"
        })

        # ✅ End monitoring
        self.monitoring_agent.end_run(status="SUCCESS")

        print("✅ Training pipeline completed")

        return {
            "cleaned_data_path": cleaned_data_path,
            "feature_metadata": fe_metadata,
            "automl_report": automl_report,
            "evaluation_report": eval_report
        }
    

