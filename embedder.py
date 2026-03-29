# This file converts text chunks into embeddings.
# An embedding is a list of numbers that captures the meaning of the text.

import ollama

from chunker import chunk_text
from extract import extract_text


def embed_chunks(chunks):
    # This function loops through each chunk and creates one embedding vector for it.
    embedded = []
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        embedded.append({
            "id": str(i),
            "text": chunk,
            "vector": response["embedding"]
        })
        # Print progress so large PDFs do not look stuck.
        if i % 50 == 0:
            print(f"Embedded {i}/{len(chunks)} chunks...")
    return embedded


if __name__ == "__main__":
    # This block is for testing the embedding step on a sample PDF.
    text = extract_text("data/sample.pdf")
    chunks = chunk_text(text)
    embedded = embed_chunks(chunks)

    print(f"\nTotal embedded: {len(embedded)}")
    print(f"Vector size: {len(embedded[0]['vector'])}")
    print(f"Sample vector (first 5 numbers): {embedded[0]['vector'][:5]}")
