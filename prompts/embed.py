# embeddings/embed.py

import os
from typing import Literal

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

def get_embed_function(model: Literal["openai", "huggingface"] = "openai") -> Embeddings:
    if model == "openai":
        return OpenAIEmbeddings()

    if model == "huggingface":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    raise ValueError(f"Unsupported embedding model: {model}")

