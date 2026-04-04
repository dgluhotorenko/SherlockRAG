"""Build (or rebuild) the ChromaDB vector store from text files in the data directory."""

import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import DATA_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from rag_chain import get_embedding_function


def ingest():
    print(f"Loading documents from {DATA_DIR} ...")

    documents = []
    for file_path in sorted(Path(DATA_DIR).glob("**/*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        exit(f"No documents found in {DATA_DIR}.")

    print(f"Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("Removed existing vector store.")

    print("Creating embeddings and saving to ChromaDB ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Vector store saved to {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
