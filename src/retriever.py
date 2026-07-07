import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # pulls GROQ_API_KEY from .env, same pattern as any API-key project (Month 3)


def load_vector_collection(config):
    """
    Loads the Chroma collection that R3's embed_and_store_chunks() built
    and persisted to disk. This is the "trained artifact" equivalent of
    predict.py loading model.pkl in Bike Demand — nothing is recomputed,
    we just open what already exists on disk.

    Also loads the SAME embedding model used at ingestion time (R3 Q7):
    a question can only be compared against stored vectors if it's
    encoded into the exact same vector space.
    """
    persist_path = config["vector_store"]["persist_path"]
    collection_name = config["vector_store"]["collection_name"]
    model_name = config["embedding"]["model_name"]

    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_collection(name=collection_name)
    model = SentenceTransformer(model_name)

    return collection, model


def retrieve_chunks(question, collection, model, top_k):
    """
    R4 core retrieval step.

    1. Encode the question with the same embedding model used in R3.
    2. Ask Chroma for the top_k most similar stored chunks (by vector distance).
    3. Return both the chunk text AND its metadata (source, page) — we need
       metadata later so the final answer can point back to which file/page
       it came from, same reason metadata was never optional in R2/R3.

    Returns a list of dicts: {"text": ..., "metadata": {...}}
    """
    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    documents = results["documents"][0]   # list of chunk texts
    metadatas = results["metadatas"][0]   # list of matching metadata dicts

    chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(documents, metadatas)
    ]
    return chunks


def build_context_block(chunks):
    """
    Turns retrieved chunks into the single block of text that gets
    inserted into the prompt (R4 Q6). Each chunk is labeled with its
    source/page so the LLM can cite where information came from —
    this is what makes "grounding" (Q4) concrete: the model only sees
    this text, nothing from its own training data about these documents.
    """
    parts = []
    for chunk in chunks:
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page", "?")
        parts.append(f"[Source: {source}, Page: {page}]\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def build_qa_chain(config):
    """
    Builds the actual LangChain chain: Prompt Template -> Groq LLM -> Parser.

    This is the NEW piece compared to what you've done before: instead of
    manually calling the Groq API and string-formatting a prompt yourself
    (Month 3 style), LangChain's LCEL "|" syntax composes these steps into
    one reusable object you can just call with .invoke(...).

    The prompt explicitly instructs the model to answer ONLY from the
    provided context — this is the grounding instruction, not just a
    side-effect of what data we happened to send.
    """
    llm = ChatGroq(
        model=config["llm"]["model_name"],
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about business "
         "documents (invoices, receipts, contracts). Answer ONLY using the "
         "context provided below. If the answer is not in the context, say "
         "you don't have enough information — do not guess or use outside "
         "knowledge."),
        ("human",
         "Context:\n{context}\n\nQuestion: {question}"),
    ])

    parser = StrOutputParser()

    chain = prompt | llm | parser
    return chain


def answer_question(question, config, top_k=4):
    """
    Full R4 pipeline, entry point for this module.

    Question -> encode -> retrieve top_k chunks -> build context block
    -> run through the LangChain chain -> grounded answer string.

    top_k defaults to 4 as a starting point; this will become a config
    value once we wire in MLflow tracking in R6 (so different k values
    can be compared as experiments, same idea as tuning hyperparameters
    in Bike Demand's LightGBM training).
    """
    collection, model = load_vector_collection(config)
    chunks = retrieve_chunks(question, collection, model, top_k)

    if not chunks:
        return "No relevant documents found in the vector store."

    context = build_context_block(chunks)
    chain = build_qa_chain(config)

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return answer