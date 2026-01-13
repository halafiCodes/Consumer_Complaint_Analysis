import os
from typing import List, Tuple, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize persistent vector store and embedding model
_persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vector_store"))
_client = chromadb.PersistentClient(path=_persist_dir)
_collection = _client.get_collection(name="complaints")
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_top_k(question: str, k: int = 5) -> List[Dict]:
    """
    Retrieve top-k complaint chunks for a user question using query embeddings.
    Returns a list of dicts with text and metadata.
    """
    query_emb = _embed_model.encode([question], show_progress_bar=False).tolist()
    results = _collection.query(query_embeddings=query_emb, n_results=k)

    retrieved: List[Dict] = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "text": text,
            "metadata": metadata,
        })
    return retrieved


def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build a prompt for the LLM using retrieved chunks as context.
    """
    context_text = "\n\n".join([f"- {c['text']}" for c in retrieved_chunks])
    prompt = f"""
You are a financial analyst assistant for CrediTrust Financial.
Answer questions about customer complaints using ONLY the context below.
Do not make up information.

Context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt.strip()


def _get_generator():
    """Return a lightweight CPU text generation pipeline."""
    return pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1,  # CPU
    )


def answer_question(question: str, top_k: int = 5, max_tokens: int = 64) -> Tuple[str, List[Dict]]:
    """
    Retrieve top-k chunks and generate answer using a lightweight LLM.
    Returns the answer and retrieved chunks.
    """
    retrieved_chunks = retrieve_top_k(question, k=top_k)
    prompt = build_prompt(question, retrieved_chunks)

    generator = _get_generator()
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
    answer = output[0]["generated_text"]
    return answer, retrieved_chunks
