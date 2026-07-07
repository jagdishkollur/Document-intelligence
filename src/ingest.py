import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


def extract_text_from_pdf(file_path):
    """
    Reads a PDF file page by page and returns a list of
    (text, page_number) tuples — one entry per page.

    We keep page numbers separate (not joined into one giant string)
    because page number is required metadata, per Q3 in R2 recall check.
    """
    reader = PdfReader(file_path)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:  # some pages can be blank/scanned-image-only -> None
            pages_text.append((text, page_number))

    return pages_text


def extract_text_from_txt(file_path):
    """
    Reads a .txt file directly. No parsing needed — the bytes
    already are the text (see R2 Q2/Q5).
    Returned as a list with one entry so the shape matches
    extract_text_from_pdf's output: (text, page_number).
    .txt files have no concept of "pages", so we use page_number=1.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [(text, 1)]


def load_documents(config):
    """
    Walks data/raw/ (path comes from config, per R1 Q7 — never hardcode it),
    extracts text from every file, and attaches metadata to each page.

    Returns a list of dicts, one per page/chunk-source:
        {
            "text": "...",
            "metadata": {
                "source": "invoice_march.pdf",
                "page": 2
            }
        }

    If one file fails, it is logged and skipped — the other files
    must still be processed (R2 Q6).
    """
    raw_docs_path = config["data"]["raw_docs_path"]
    documents = []

    if not os.path.exists(raw_docs_path):
        print(f"Warning: {raw_docs_path} does not exist. No documents to process.")
        return documents

    for filename in os.listdir(raw_docs_path):
        file_path = os.path.join(raw_docs_path, filename)

        try:
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
                        "page": page_number
                    }
                })

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue  # move on to the next file, don't crash the batch

    print(f"Loaded {len(documents)} page(s)/section(s) from {raw_docs_path}.")
    return documents


def chunk_documents(documents, config):
    """
    R3 — Splits each page's text into smaller overlapping chunks.

    Uses RecursiveCharacterTextSplitter, which tries to split on
    paragraph breaks first, then lines, then words, then characters
    — only falling back to a harder cut if the piece is still too big
    (R3 Q1). chunk_overlap repeats a bit of the previous chunk into the
    next one so a fact/sentence straddling a cut point isn't lost (Q2).

    Each resulting chunk inherits the SAME metadata (source, page) as
    the page it came from — a page can produce multiple chunks, and
    every one of them needs to stay traceable back to its file (Q3/Q8).

    Returns a list of dicts:
        {
            "text": "<chunk text>",
            "metadata": {"source": ..., "page": ...}
        }
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

        for piece in text_pieces:
            chunks.append({
                "text": piece,
                "metadata": doc["metadata"],  # copied as-is onto every chunk
            })

    print(f"Split {len(documents)} page(s) into {len(chunks)} chunk(s).")
    return chunks


def embed_and_store_chunks(chunks, config):
    """
    R3 — Generates an embedding vector for every chunk and stores
    (text + vector + metadata) together in a persistent Chroma collection.

    Model: all-MiniLM-L6-v2, loaded locally via sentence-transformers
    — no API key needed (per R2 side-topic clarification).

    Identical chunk text always produces the identical vector; what
    keeps otherwise-identical chunks distinguishable is the metadata
    stored alongside the vector, not the vector itself (R3 Q7).

    Design choice for this session: the collection is DELETED and
    rebuilt from scratch on every call — simplest correct behavior
    for now. This means re-running ingestion on the same data/raw/
    folder is safe and won't duplicate chunks, but it also means any
    previously ingested documents whose files are no longer in
    data/raw/ will be dropped from the store. Append-only / de-duplication
    logic is a deliberate deferral to a future R-session, not an oversight.
    """
    model_name = config["embedding"]["model_name"]
    persist_path = config["vector_store"]["persist_path"]
    collection_name = config["vector_store"]["collection_name"]

    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=persist_path)

    # Wipe and rebuild: drop the old collection if it exists, then recreate it.
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # collection didn't exist yet — nothing to delete

    collection = client.create_collection(name=collection_name)

    if not chunks:
        print("No chunks to embed. Skipping storage step.")
        return 0

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"Generating embeddings for {len(texts)} chunk(s) ...")
    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunk(s) in Chroma collection '{collection_name}'.")
    return len(chunks)


def run_ingestion(config):
    """
    Entry point called by run_pipeline.py.

    Full R3 pipeline:
        load_documents()      -> pages with metadata (R2)
        chunk_documents()     -> smaller overlapping chunks (R3)
        embed_and_store_chunks() -> vectors + text + metadata in Chroma (R3)

    Returns the number of chunks actually stored, so run_pipeline.py's
    printed "chunks stored" count is now accurate (previously a
    page-count placeholder).
    """
    documents = load_documents(config)
    chunks = chunk_documents(documents, config)
    num_stored = embed_and_store_chunks(chunks, config)
    return num_stored