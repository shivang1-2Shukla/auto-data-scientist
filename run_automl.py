from orchestrator.orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator(task_type="regression")

    result = orchestrator.run_training_pipeline(
        raw_data_path="data/sample.csv",
        target_column="age"
    )

    print("📦 Final pipeline result:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
