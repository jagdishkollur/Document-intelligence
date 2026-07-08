# Document Intelligence System

An automated RAG (Retrieval-Augmented Generation) pipeline that lets users ask natural-language questions across a growing collection of documents (PDFs, invoices, contracts, notes) — with new documents ingested automatically the moment they're dropped into a watched folder.

> **Portfolio note:** This project demonstrates taking an ML/NLP capability (RAG-based Q&A) out of a notebook and turning it into a fully automated system — an API, a persistent vector store with incremental updates, a no-code automation layer, and a chat UI — with zero manual intervention required once deployed.

---

## What it does

1. Documents (PDF/TXT) are dropped into a Google Drive folder.
2. An automation workflow detects the new file, downloads it, and sends it to a running API.
3. The API extracts text, splits it into chunks, embeds those chunks, and stores them in a persistent vector database — **skipping unchanged files and only re-processing new or edited ones.**
4. A Telegram message confirms success (or reports failure) for every file processed.
5. Users ask questions through a chat-style web interface, which retrieves the most relevant chunks and asks an LLM to answer using only that retrieved context.
6. Every ingestion run is logged (chunk counts, timing, model config) for auditability.

---

## Architecture


```
┌───────────────────────────┐
│   Google Drive folder     │
│ (Document-Intelligence-   │
│         Inbox)            │
└─────────────┬─────────────┘
              │ new file detected
              │ (polled every minute)
              ▼
┌────────────────────────────┐
│         n8n Cloud          │
│  1. Google Drive Trigger   │
│  2. Download File          │
│  3. HTTP Request  ─────────┼───┐
│  4. Telegram (success or   │   │
│     failure branch)        │   │
└────────────────────────────┘   │
                                  │ POST file bytes
                                  │ over public internet
                                  ▼
                     ┌───────────────────────────┐
                     │       ngrok tunnel        │
                     │  (public HTTPS → local)   │
                     └─────────────┬─────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────┐
│                 FastAPI (serve.py)                        │
│  POST /ingest-file  — save file + trigger ingestion       │
│  POST /ingest       — trigger ingestion manually          │
│  POST /query        — ask a question                      │
│  GET  /documents    — list indexed files                  │
│  GET  /health       — liveness check                      │
└───────────┬───────────────┬─────────────────────────────┬─┘
             │               │                            │
             ▼               ▼                            ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────┐
│  src/ingest.py        │ │ src/retriever.py      │ │  src/tracker.py     │
│                       │ │                       │ │                     │
│ - extract text        │ │ - load Chroma         │ │ - MLflow logging    │
│   (PDF/TXT)           │ │   collection          │ │   (params + metrics │
│ - hash-based          │ │ - build retriever     │ │   per ingestion run)│
│   skip/new/changed    │ │ - LangChain LCEL      │ │                     │
│   classification      │ │   chain               │ │                     │
│ - chunk (Recursive    │ │ - Groq LLM answer     │ │                     │
│   CharacterText       │ │                       │ │                     │
│   Splitter)           │ └───────────────────────┘ └─────────────────────┘
│ - embed (sentence-    │
│   transformers,       │
│   all-MiniLM-L6-v2)   │
│ - store in Chroma     │
│   (persistent, local) │
└──────────┬────────────┘
            ▼
┌────────────────────────┐
│   Chroma vector store  │
│    (vector_store/)     │
└────────────────────────┘

┌────────────────────────────┐
│    Streamlit UI (app.py)   │
│  - chat interface          │
│  - calls POST /query       │
│  - session-persistent chat │
│    history                 │
└────────────────────────────┘
```


### Two ways new content gets ingested

| Path | Trigger | Ingestion happens |
|---|---|---|
| **Automated** | File dropped in the watched Google Drive folder | n8n detects it, downloads it, POSTs to `/ingest-file`, which saves it to `data/raw/` **and** runs ingestion in the same step |
| **Manual** | File copied directly into `data/raw/` on the local machine | Nothing watches this folder automatically — ingestion must be triggered by calling `POST /ingest` yourself |

Both paths run the exact same underlying `run_ingestion()` logic — the automation layer is just a different way of invoking it.

### Incremental ingestion (no wasteful full rebuilds)

Every file is classified by comparing a SHA-256 hash of its content against what's already stored:

- **SKIP** — filename exists, hash matches → nothing happens
- **NEW** — filename never seen before → processed and added
- **CHANGED** — filename exists, hash differs (file was edited and re-dropped with the same name) → old chunks for that file are deleted, then it's re-processed

This means re-running ingestion (or re-uploading the same file by mistake) is cheap and safe — it will not duplicate data or destroy the whole collection.

---

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) — local, free, no API key |
| Vector database | Chroma (persistent, local) |
| RAG orchestration | LangChain (LCEL) |
| API | FastAPI + uvicorn |
| Automation | n8n Cloud |
| Experiment tracking | MLflow (SQLite backend) |
| UI | Streamlit |
| Tunneling (local dev only) | ngrok |
| Config | YAML |

---

## Project structure

```
document-intelligence/
├── data/
│   └── raw/              ← documents land here (via /ingest-file or manually)
├── vector_store/         ← Chroma DB persisted here
├── src/
│   ├── ingest.py         ← parse, chunk, embed, hash-based incremental storage
│   ├── retriever.py      ← load Chroma + RAG chain (LangChain + Groq)
│   └── tracker.py        ← MLflow logging per ingestion run
├── serve.py              ← FastAPI endpoints (/query, /documents, /health, /ingest, /ingest-file)
├── app.py                ← Streamlit chat UI
├── run_pipeline.py       ← pipeline runner
├── config.yaml           ← all config values (paths, models, chunk size, API base URL)
├── requirements.txt      ← dependencies
└── README.md             ← this file
```

---

## Running it locally

Three processes run independently, each in its own terminal:

**1. Start the API**
```
uvicorn serve:app --host 0.0.0.0 --port 8000
```
> Note: on Windows with Python 3.14, do **not** use `--reload` — it causes a multiprocessing spawn error.

**2. Start the UI**
```
streamlit run app.py
```
Opens a browser tab where you can ask questions directly.

**3. (Optional) Enable automated ingestion from Google Drive**

Only needed if you want the Drive-folder automation, not required to use the API or UI:
```
ngrok http 8000
```
Copy the HTTPS URL ngrok prints and update the n8n workflow's HTTP Request node with it (see **Known Limitations** below).

**Manual ingestion** (if you copied files into `data/raw/` by hand instead of using Drive):
```
curl.exe -X POST http://localhost:8000/ingest
```

---

## Known limitations

- **ngrok URL is temporary.** The free-tier ngrok URL changes every time the tunnel is restarted, and the n8n workflow's HTTP Request node has it hardcoded. This is a known, accepted limitation for a local-dev/portfolio demo — a production deployment would replace this with a permanently hosted API (no tunnel needed) or a paid static ngrok domain.
- **Only text-based PDFs are supported.** Scanned/image-only PDFs have no extractable text layer and would silently produce zero chunks — OCR support would be a future enhancement.
- **No one-click startup yet.** Running this currently requires manually starting `uvicorn` and `streamlit` in separate terminals. A packaged startup script or full cloud deployment would be needed to make this usable by a non-technical end user without developer assistance.
- **Google Drive filenames can differ slightly from local filenames** (e.g. underscores may be stripped on sync) — since ingestion keys off exact filename strings, a file that looks "the same" to a human may be tracked as a distinct file if its name changes between environments.

---

## Automation failure handling

The n8n workflow notifies via Telegram on **both** outcomes:
- ✅ Success — filename, chunks stored, documents processed
- ❌ Failure — filename and error status, if the HTTP request to the API fails (e.g. the API or tunnel is down)

This was a deliberate addition beyond the original design — silent failure isn't acceptable for a system pitched as "automated and self-updating."

---

## Freelance / portfolio positioning

This project demonstrates the ability to take an ML capability out of a notebook and turn it into a real automated system: an API, a persistent and incrementally-updating data store, a no-code automation layer connecting cloud storage to a local service, and a usable chat interface — with experiment tracking for auditability throughout.
