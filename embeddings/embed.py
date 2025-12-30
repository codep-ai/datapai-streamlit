# embeddings/embed.py
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings

# Instantiate once
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> List[List[float]]:
    return embedding_model.embed_documents(texts)

def embed_query(query: str) -> List[float]:
    return embedding_model.embed_query(query)

