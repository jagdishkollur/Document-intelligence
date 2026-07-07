import yaml
from src.ingest import run_ingestion
from src.tracker import timed_ingestion


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print("Config loaded.")
    print(f"Watching folder: {config['data']['raw_docs_path']}")

    print("\nStarting ingestion...")

    # Changed from calling run_ingestion(config) directly (R1-R3 behavior)
    # to timed_ingestion(), which runs the same ingestion function but
    # also measures duration and logs params/metrics to MLflow (R6) —
    # run_pipeline.py itself doesn't need to know MLflow exists.
    num_chunks = timed_ingestion(run_ingestion, config)

    print(f"Ingestion complete. {num_chunks} chunks stored in Chroma.")
    print("Run details logged to MLflow — view with: mlflow ui")


if __name__ == "__main__":
    main()