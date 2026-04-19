from orchestrator.orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator(task_type="regression")

    result = orchestrator.run_training_pipeline(
        raw_data_path="data/sample.csv",
        target_column="age"
    )

    print("\n📦 Final pipeline result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\n" + "="*50)
    print("🎉 Project execution complete!")
    print("To start the API and serve predictions, run the following command:")
    print("    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
