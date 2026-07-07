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
    num_chunks, num_documents = timed_ingestion(run_ingestion, config)

    print(f"Ingestion complete. {num_chunks} chunk(s) stored across {num_documents} document(s).")
    print("Run details logged to MLflow — view with: mlflow ui")


if __name__ == "__main__":
    main()