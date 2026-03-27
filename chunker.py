from langchain_text_splitters import RecursiveCharacterTextSplitter
from extract import extract_text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    text = extract_text("data/sample.pdf")
    chunks = chunk_text(text)
    
    print(f"Total chunks: {len(chunks)}")
    print(f"\n--- Chunk 1 ---\n{chunks[0]}")
    print(f"\n--- Chunk 2 ---\n{chunks[1]}")
    print(f"\n--- Chunk 3 ---\n{chunks[2]}")