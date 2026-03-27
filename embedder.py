import ollama
from extract import extract_text
from chunker import chunk_text

def embed_chunks(chunks):
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
        if i % 50 == 0:
            print(f"Embedded {i}/{len(chunks)} chunks...")
    return embedded

if __name__ == "__main__":
    text = extract_text("data/sample.pdf")
    chunks = chunk_text(text)
    embedded = embed_chunks(chunks)

    print(f"\nTotal embedded: {len(embedded)}")
    print(f"Vector size: {len(embedded[0]['vector'])}")
    print(f"Sample vector (first 5 numbers): {embedded[0]['vector'][:5]}")