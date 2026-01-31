from orchestrator.orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator(task_type="regression")

    result = orchestrator.run_training_pipeline(
        data_path="data/processed/cleaned.csv",
        target_column="age"
    )

    print("📦 Final pipeline result:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()