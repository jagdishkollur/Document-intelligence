import time
import mlflow


def start_tracking(config):
    """
    Points MLflow at the sqlite backend and selects/creates the
    experiment, using values from config.yaml. Must run BEFORE
    mlflow.start_run() is called.
    """
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_ingestion_run(config, num_chunks, num_documents, duration_seconds):
    """
    Logs one ingestion run as a single MLflow run.

    PARAMS (settings configured BEFORE running):
        embedding model name, chunk size, chunk overlap.
    METRICS (outcomes MEASURED after running):
        chunks stored, documents processed, ingestion duration.

    Everything below must stay inside the with mlflow.start_run():
    block -- that's what ties every log_param/log_metric call to one
    specific run ID.
    """
    with mlflow.start_run():
        mlflow.log_param("embedding_model", config["embedding"]["model_name"])
        mlflow.log_param("chunk_size", config["chunking"]["chunk_size"])
        mlflow.log_param("chunk_overlap", config["chunking"]["chunk_overlap"])

        mlflow.log_metric("num_chunks_stored", num_chunks)
        mlflow.log_metric("num_documents_processed", num_documents)
        mlflow.log_metric("ingestion_duration_seconds", duration_seconds)


def timed_ingestion(ingestion_fn, config):
    """
    Runs the given ingestion function, measures duration, logs
    everything to MLflow in one call.

    ingestion_fn (run_ingestion) now returns a tuple:
        (num_chunks_stored, num_documents_processed)
    num_documents_processed is a real count as of the incremental
    ingestion update -- no longer a hardcoded 0.

    Returns the same (num_chunks, num_documents) tuple back to the
    caller (run_pipeline.py or the /ingest API endpoint), so they can
    report both numbers without needing MLflow-specific code.
    """
    start_tracking(config)

    start_time = time.time()
    num_chunks, num_documents = ingestion_fn(config)
    duration_seconds = time.time() - start_time

    log_ingestion_run(
        config,
        num_chunks=num_chunks,
        num_documents=num_documents,
        duration_seconds=duration_seconds,
    )

    return num_chunks, num_documents