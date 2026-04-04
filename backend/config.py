import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BACKEND_DIR / ".." / "data"))).resolve()
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BACKEND_DIR / "chroma_db"))).resolve()

PROMPT_TEMPLATE = """You are a precise and factual assistant answering questions about the book 'The Adventures of Sherlock Holmes'.
Use ONLY the provided context below to answer the user's question.
Be concise and answer directly. Do not add any conversational phrases or apologies.
If the information to answer the question is not in the context, respond with "The answer is not available in the provided text."

Context:
{context}

Question: {question}

Answer:"""
