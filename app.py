# This is the main Streamlit app.
# It handles the full flow:
# 1. Read the PDF
# 2. Split it into chunks
# 3. Turn chunks into embeddings
# 4. Store them in ChromaDB
# 5. Retrieve relevant chunks for a question
# 6. Ask Groq to answer using only those chunks

import os
import re

import chromadb
import fitz
import ollama
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load values from the .env file so the app can read the Groq API key.
load_dotenv()

# These settings control which models and retrieval values the app uses.
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
INITIAL_RETRIEVAL_K = 8
FINAL_CONTEXT_K = 4

# Basic Streamlit page setup.
st.set_page_config(page_title="PDF Chatbot", page_icon="PDF")
st.title("PDF Chatbot")
st.caption("Powered by Groq + ChromaDB")


def extract_text(pdf_file):
    # This function reads the uploaded PDF and returns all page text as one string.
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []

    for page in doc:
        page_text = page.get_text()
        # Cleaning extra whitespace makes retrieval more consistent.
        page_text = re.sub(r"[ \t]+", " ", page_text)
        page_text = re.sub(r"\n{3,}", "\n\n", page_text)
        pages.append(page_text.strip())

    return "\n\n".join(page for page in pages if page)


def chunk_text(text):
    # This function splits long text into smaller overlapping parts.
    # Overlap helps when the answer sits between two chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def build_vectorstore(chunks):
    # This function creates a ChromaDB collection and stores one embedding per chunk.
    client = chromadb.PersistentClient(path="db")

    try:
        # Delete the old collection so the new PDF replaces previous data.
        client.delete_collection("pdf_chunks")
    except Exception:
        pass

    collection = client.get_or_create_collection(name="pdf_chunks")

    for i, chunk in enumerate(chunks):
        # Ollama creates a numeric vector that represents the chunk meaning.
        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=chunk
        )
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[response["embedding"]]
        )
    return collection


def embed_query(text):
    # This function turns the user's question into an embedding vector.
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]


def tokenize(text):
    # This function creates simple keywords from text for light reranking.
    return {
        token for token in re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
        if token not in {"what", "which", "when", "where", "this", "that", "with", "from", "have"}
    }


def rerank_chunks(query, documents, distances):
    # This function improves retrieval by mixing vector similarity with keyword overlap.
    query_terms = tokenize(query)
    scored_chunks = []

    for doc, distance in zip(documents, distances):
        doc_lower = doc.lower()
        overlap = sum(1 for term in query_terms if term in doc_lower)
        exact_phrase_bonus = 3 if query.lower() in doc_lower else 0

        # We mix keyword overlap with the vector distance so exact matches
        # are less likely to be missed.
        score = overlap + exact_phrase_bonus - float(distance)
        scored_chunks.append((score, doc))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_chunks[:FINAL_CONTEXT_K]]


def retrieve(query, collection, initial_k=INITIAL_RETRIEVAL_K):
    # This function finds the most useful chunks for the user's question.
    query_vector = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=initial_k,
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    return rerank_chunks(query, documents, distances)


def generate_answer(query, collection):
    # This function gets relevant chunks and asks Groq to write the final answer.
    chunks = retrieve(query, collection)
    context = "\n\n".join(chunks)
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
        # Import here so the app can still open even if the package is missing.
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


# Streamlit session state keeps data available between interactions.
if "collection" not in st.session_state:
    st.session_state.collection = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar is used to upload and process a PDF.
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Extracting text..."):
                text = extract_text(uploaded_file)
            with st.spinner("Chunking..."):
                chunks = chunk_text(text)
            with st.spinner(f"Embedding {len(chunks)} chunks..."):
                st.session_state.collection = build_vectorstore(chunks)
            st.success(f"Ready! {len(chunks)} chunks indexed.")
            st.session_state.messages = []

# Show previous chat messages on the screen.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If no PDF is processed yet, guide the user.
if st.session_state.collection is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    # Once a PDF is ready, let the user ask questions about it.
    query = st.chat_input("Ask a question about your PDF...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer(query, st.session_state.collection)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
