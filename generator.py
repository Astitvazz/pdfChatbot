import ollama
from retriever import retrieve

def generate_answer(query, k=3):
    chunks = retrieve(query, k)
    
    context = "\n\n".join(chunks)
    
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't find that in the document."

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]

if __name__ == "__main__":
    print("PDF Chatbot ready. Type 'quit' to exit.\n")
    
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            break
        answer = generate_answer(query)
        print(f"\nBot: {answer}\n")