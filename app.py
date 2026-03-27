import streamlit as st
import ollama
import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("PDF Chatbot")
st.caption("Powered by Ollama + ChromaDB")

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def build_vectorstore(chunks):
    client = chromadb.PersistentClient(path="db")
    
    try:
        client.delete_collection("pdf_chunks")
    except:
        pass
    
    collection = client.get_or_create_collection(name="pdf_chunks")
    
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[response["embedding"]]
        )
    return collection

def retrieve(query, collection, k=3):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=k
    )
    return results["documents"][0]

def generate_answer(query, collection):
    chunks = retrieve(query, collection)
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't find that in the document."

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

if "collection" not in st.session_state:
    st.session_state.collection = None

if "messages" not in st.session_state:
    st.session_state.messages = []

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.collection is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
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