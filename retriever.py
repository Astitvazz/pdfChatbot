# This file is responsible for finding which stored chunks best match a question.

import re

import chromadb
import ollama

# These settings control how many chunks are searched and returned.
EMBED_MODEL = "nomic-embed-text"
INITIAL_RETRIEVAL_K = 8
FINAL_CONTEXT_K = 4


def get_collection():
    # This function opens the saved ChromaDB collection from disk.
    client = chromadb.PersistentClient(path="db")
    return client.get_or_create_collection(name="pdf_chunks")


def embed_query(text):
    # This function converts the user's question into an embedding vector.
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]


def tokenize(text):
    # This function extracts simple keywords used for reranking.
    return {
        token for token in re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
        if token not in {"what", "which", "when", "where", "this", "that", "with", "from", "have"}
    }


def rerank_chunks(query, documents, distances):
    # This function improves search results by mixing meaning-based search
    # with simple keyword matching.
    query_terms = tokenize(query)
    scored_chunks = []

    for doc, distance in zip(documents, distances):
        doc_lower = doc.lower()
        overlap = sum(1 for term in query_terms if term in doc_lower)
        exact_phrase_bonus = 3 if query.lower() in doc_lower else 0
        score = overlap + exact_phrase_bonus - float(distance)
        scored_chunks.append((score, doc))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_chunks[:FINAL_CONTEXT_K]]


def retrieve(query, k=FINAL_CONTEXT_K):
    # This function fetches candidate chunks from ChromaDB,
    # reranks them, and returns the best few.
    collection = get_collection()
    query_vector = embed_query(query)

    # We fetch more candidates first, then keep only the best matching ones.
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=max(INITIAL_RETRIEVAL_K, k),
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    reranked = rerank_chunks(query, documents, distances)
    return reranked[:k]


if __name__ == "__main__":
    # This block is for quick local testing.
    query = "what is a black hole"
    chunks = retrieve(query)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)
