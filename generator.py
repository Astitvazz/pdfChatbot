# This file handles the final answer generation step.
# It retrieves useful chunks first, then sends them to Groq.

import os

from dotenv import load_dotenv

from retriever import retrieve

# Load environment variables such as the Groq API key.
load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def generate_answer(query, k=4):
    # This function gets the most relevant chunks and asks the LLM to answer.
    chunks = retrieve(query, k=k)
    context = "\n\n".join(chunks)

    # The model sees only these retrieved chunks, so retrieval quality matters a lot.
    prompt = f"""You are a helpful assistant for question answering over a PDF.
Use the context below to answer the question as clearly as possible.
If the answer is partly present across multiple chunks, combine them carefully.
Only say "I don't find that in the document." when the answer is truly missing from the context.

Context:
{context}

Question: {query}

Answer:"""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Missing GROQ_API_KEY. Add it to your .env file and try again."

    try:
        # Import here so the script can fail gracefully if Groq is not installed.
        from groq import Groq
    except ImportError:
        return "Groq package is not installed. Run: pip install groq"

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # This block lets you chat with the bot in the terminal.
    print("PDF Chatbot ready. Type 'quit' to exit.\n")

    while True:
        query = input("You: ")
        if query.lower() == "quit":
            break
        answer = generate_answer(query)
        print(f"\nBot: {answer}\n")
