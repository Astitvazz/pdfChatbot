# This file is only responsible for splitting long PDF text into smaller chunks.
# Chunks are easier to store, search, and send to the language model.

from langchain_text_splitters import RecursiveCharacterTextSplitter

from extract import extract_text

# These values control chunk length and how much one chunk overlaps the next.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def chunk_text(text):
    # This function takes one long string and breaks it into overlapping chunks.
    # Larger chunks with overlap help keep answers from being split apart.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    # This block is just for local testing from the terminal.
    text = extract_text("data/sample.pdf")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")
    print(f"\n--- Chunk 1 ---\n{chunks[0]}")
    print(f"\n--- Chunk 2 ---\n{chunks[1]}")
    print(f"\n--- Chunk 3 ---\n{chunks[2]}")
