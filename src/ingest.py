import os
import hashlib
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


def extract_text_from_pdf(file_path):
    """
    Reads a PDF file page by page and returns a list of
    (text, page_number) tuples — one entry per page.
    """
    reader = PdfReader(file_path)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages_text.append((text, page_number))

    return pages_text


def extract_text_from_txt(file_path):
    """
    Reads a .txt file directly. Returned as a list with one entry
    so the shape matches extract_text_from_pdf's output.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [(text, 1)]


def compute_file_hash(file_path):
    """
    Hashes a file's raw bytes (SHA-256). Used to detect whether a
    file's content has changed since it was last ingested, even if
    the filename stayed the same.
    """
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_existing_source_hashes(collection):
    """
    Reads all chunk metadata currently in the Chroma collection and
    builds a {filename: file_hash} map — one entry per source file.
    Returns an empty dict if the collection is empty (first run ever).
    """
    all_items = collection.get(include=["metadatas"])
    hashes = {}
    for meta in all_items.get("metadatas", []):
        source = meta.get("source")
        file_hash = meta.get("file_hash")
        if source and file_hash:
            hashes[source] = file_hash
    return hashes


def load_documents(config, existing_hashes):
    """
    Walks data/raw/, classifies each file as NEW, CHANGED, or SKIPPED
    based on content hash, and extracts text only for files that need
    (re)processing.

    NEW      = filename never seen before in the collection
    CHANGED  = filename exists, but file content hash differs
               (old chunks for this file must be deleted before
               re-adding, handled later in embed_and_store_chunks)
    SKIPPED  = filename exists and hash matches -> do nothing

    Returns:
      documents: list of {"text": ..., "metadata": {...}} for
                 NEW + CHANGED files only
      changed_sources: set of filenames whose OLD chunks must be
                 deleted before new ones are stored
    """
    raw_docs_path = config["data"]["raw_docs_path"]
    documents = []
    changed_sources = set()

    if not os.path.exists(raw_docs_path):
        print(f"Warning: {raw_docs_path} does not exist. No documents to process.")
        return documents, changed_sources

    for filename in os.listdir(raw_docs_path):
        file_path = os.path.join(raw_docs_path, filename)

        try:
            file_hash = compute_file_hash(file_path)

            if filename in existing_hashes and existing_hashes[filename] == file_hash:
                print(f"Skipping unchanged file: {filename}")
                continue

            if filename in existing_hashes:
                print(f"File changed, will re-ingest: {filename}")
                changed_sources.add(filename)
            else:
                print(f"New file, will ingest: {filename}")

            if filename.lower().endswith(".pdf"):
                pages = extract_text_from_pdf(file_path)
            elif filename.lower().endswith(".txt"):
                pages = extract_text_from_txt(file_path)
            else:
                print(f"Skipping unsupported file type: {filename}")
                continue

            for text, page_number in pages:
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_number,
                        "file_hash": file_hash,
                    }
                })

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print(f"Loaded {len(documents)} page(s)/section(s) needing (re)ingestion.")
    return documents, changed_sources


def chunk_documents(documents, config):
    """
    Splits each page's text into smaller overlapping chunks using
    RecursiveCharacterTextSplitter. Each chunk inherits the same
    metadata (source, page, file_hash) as the page it came from,
    plus a chunk_index used later to build a stable, unique chunk ID.
    """
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for doc in documents:
        text_pieces = splitter.split_text(doc["text"])

        for chunk_index, piece in enumerate(text_pieces):
            chunks.append({
                "text": piece,
                "metadata": {**doc["metadata"], "chunk_index": chunk_index},
            })

    print(f"Split {len(documents)} page(s) into {len(chunks)} chunk(s).")
    return chunks


def get_collection(config):
    """
    Opens (or creates, on first run) the persistent Chroma collection.
    Unlike the old version, this NEVER deletes the collection — that
    is what makes incremental ingestion possible.
    """
    persist_path = config["vector_store"]["persist_path"]
    collection_name = config["vector_store"]["collection_name"]

    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def embed_and_store_chunks(chunks, config, collection, changed_sources):
    """
    Embeds and stores only the given chunks (already filtered to
    NEW + CHANGED files by load_documents). Deletes stale chunks for
    any CHANGED file first, so edited files don't end up duplicated
    alongside their old version.

    Chunk IDs are built from filename + page + chunk_index, so they
    stay unique and stable across runs -- no collisions, no
    accidental overwrites of unrelated chunks.
    """
    model_name = config["embedding"]["model_name"]

    for source in changed_sources:
        collection.delete(where={"source": source})

    if not chunks:
        print("No new or changed chunks to embed. Nothing to store.")
        return 0

    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [
        f"{m['source']}_p{m['page']}_c{m['chunk_index']}"
        for m in metadatas
    ]

    print(f"Generating embeddings for {len(texts)} chunk(s) ...")
    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} new/updated chunk(s) in Chroma collection '{collection.name}'.")
    return len(chunks)


def run_ingestion(config):
    """
    Entry point called by run_pipeline.py and the /ingest API endpoint.

    Incremental: only NEW or CHANGED files (by content hash) get
    (re)processed on each run; unchanged files are skipped entirely.

    Returns:
      (num_chunks_stored, num_documents_processed)
    num_documents_processed now reflects real files processed this
    run -- replacing the old hardcoded 0 from R6.
    """
    collection = get_collection(config)
    existing_hashes = get_existing_source_hashes(collection)

    documents, changed_sources = load_documents(config, existing_hashes)
    num_documents_processed = len({d["metadata"]["source"] for d in documents})

    chunks = chunk_documents(documents, config)
    num_stored = embed_and_store_chunks(chunks, config, collection, changed_sources)

    return num_stored, num_documents_processed