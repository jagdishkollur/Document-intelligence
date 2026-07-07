import time
import mlflow


def start_tracking(config):
    """
    Points MLflow at the sqlite backend and selects/creates the experiment,
    using values from config.yaml (never hardcoded — same reasoning as
    raw_docs_path in R1/R2).

    This must run BEFORE mlflow.start_run() is called, since it tells
    MLflow WHERE to write (tracking URI) and WHICH experiment (named
    group of runs) this run belongs to.
    """
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_ingestion_run(config, num_chunks, num_documents, duration_seconds):
    """
    Logs one ingestion run as a single MLflow run.

    PARAMS (settings that were configured BEFORE running — R6 Q2/Q5):
        - embedding model name
        - chunk size
        - chunk overlap
    These answer "what did I configure for this run?" and let you look
    back later and know exactly what produced a given result — the same
    gap we identified in Q5 (config.yaml alone can't tell you what you
    tried in the past, only what's set right now).

    METRICS (outcomes MEASURED as a result of running — R6 Q2/Q6):
        - number of chunks stored
        - number of documents processed
        - ingestion duration in seconds
    These answer "what did I get from this run?" — never the reverse of
    a param, per the Q6 gap (a param logged as a metric breaks the
    input/output distinction in the MLflow UI).

    Everything below MUST stay inside the `with mlflow.start_run():`
    block (R6 Q4) — that block is what ties every log_param/log_metric
    call to one specific run ID. Outside the block, there's no active
    run for these calls to attach to.
    """
    with mlflow.start_run():
        # --- Params: config values set BEFORE this run happened ---
        mlflow.log_param("embedding_model", config["embedding"]["model_name"])
        mlflow.log_param("chunk_size", config["chunking"]["chunk_size"])
        mlflow.log_param("chunk_overlap", config["chunking"]["chunk_overlap"])

        # --- Metrics: results MEASURED after this run happened ---
        mlflow.log_metric("num_chunks_stored", num_chunks)
        mlflow.log_metric("num_documents_processed", num_documents)
        mlflow.log_metric("ingestion_duration_seconds", duration_seconds)


def timed_ingestion(ingestion_fn, config):
    """
    Small helper: runs the given ingestion function, measures how long
    it took, and logs everything to MLflow in one call — so
    run_pipeline.py doesn't need to know about timing or logging details,
    just: "run ingestion, get num_chunks back."

    ingestion_fn is expected to be run_ingestion() from src/ingest.py,
    passed in rather than imported directly here — keeps tracker.py from
    depending on ingest.py's internals, same separation-of-concerns idea
    as predict.py not needing to know how train.py built model.pkl.
    """
    start_tracking(config)

    start_time = time.time()
    num_chunks = ingestion_fn(config)
    duration_seconds = time.time() - start_time

    # NOTE: num_documents isn't returned by run_ingestion() today — it
    # only returns num_chunks. Logging num_documents as 0 for now is a
    # placeholder; revisit if you want document-level counts tracked
    # (would require a small change to run_ingestion()'s return value).
    log_ingestion_run(
        config,
        num_chunks=num_chunks,
        num_documents=0,
        duration_seconds=duration_seconds,
    )

    return num_chunks