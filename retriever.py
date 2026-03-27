import ollama
import chromadb

def get_collection():
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name="pdf_chunks")
    return collection

def retrieve(query, k=3):
    collection = get_collection()

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )
    query_vector = response["embedding"]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    return results["documents"][0]

if __name__ == "__main__":
    query = "what is a black hole"
    chunks = retrieve(query, k=3)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)