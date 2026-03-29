# This file takes embedded chunks and stores them in ChromaDB.

import chromadb

from chunker import chunk_text
from embedder import embed_chunks
from extract import extract_text


def build_vectorstore(embedded_chunks):
    # This function writes chunk ids, text, and embeddings into the vector database.
    client = chromadb.PersistentClient(path="db")

    collection = client.get_or_create_collection(
        name="pdf_chunks"
    )

    ids = [e["id"] for e in embedded_chunks]
    texts = [e["text"] for e in embedded_chunks]
    vectors = [e["vector"] for e in embedded_chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=vectors
    )

    print(f"Stored {collection.count()} chunks in vector store")
    return collection


if __name__ == "__main__":
    # This block runs the full indexing pipeline from PDF to vector database.
    text = extract_text("data/sample.pdf")
    chunks = chunk_text(text)
    embedded = embed_chunks(chunks)
    build_vectorstore(embedded)
