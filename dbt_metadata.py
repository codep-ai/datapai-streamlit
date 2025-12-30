import json
import os
import hashlib
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Paths for manifest, metadata, and FAISS index
DBT_MANIFEST_PATH = "dbt-demo/target/manifest.json"
DBT_METADATA_PATH = "dbt_metadata.json"
FAISS_INDEX_PATH = "dbt_faiss.index"
MANIFEST_HASH_PATH = "manifest_hash.txt"

# Use a sentence-transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 🔹 Function to Compute Hash of `manifest.json`
def compute_manifest_hash():
    """Compute a hash of the dbt manifest file to detect changes."""
    with open(DBT_MANIFEST_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# 🔹 Function to Extract dbt Metadata and Update FAISS
def update_dbt_metadata():
    """Rebuild FAISS index if the dbt manifest has changed."""
    # Compute current manifest hash
    current_hash = compute_manifest_hash()

    # Check if hash file exists
    if os.path.exists(MANIFEST_HASH_PATH):
        with open(MANIFEST_HASH_PATH, "r") as f:
            saved_hash = f.read().strip()
        # If hash is unchanged, skip re-indexing
        if saved_hash == current_hash:
            print("✅ No changes in dbt manifest. Using existing FAISS index.")
            return

    # If manifest has changed, update FAISS index
    print("🔄 Changes detected in dbt manifest. Rebuilding FAISS index...")

    # Load dbt manifest.json
    with open(DBT_MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Extract model metadata
    models_metadata = []
    for model_name, model_info in manifest["nodes"].items():
        if model_info["resource_type"] == "model":
            model_metadata = {
                "model_name": model_info["name"],
                "description": model_info.get("description", "No description available"),
                "columns": {
                    col_name: col_info["description"]
                    for col_name, col_info in model_info.get("columns", {}).items()
                },
                "sql": model_info["raw_sql"],
                "dependencies": model_info["depends_on"]["nodes"],
            }
            models_metadata.append(model_metadata)

    # Save extracted metadata for RAG ingestion
    with open(DBT_METADATA_PATH, "w") as f:
        json.dump(models_metadata, f, indent=4)

    print(f"✅ Extracted dbt metadata saved to {DBT_METADATA_PATH}")

    # Convert metadata into a list of text strings for embeddings
    texts = [
        f"Model: {item['model_name']}\nDescription: {item['description']}\nColumns: {', '.join(item['columns'].keys())}"
        for item in models_metadata
    ]

    # Generate embeddings and update FAISS index
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"✅ FAISS index updated and saved to {FAISS_INDEX_PATH}")

    # Save new manifest hash
    with open(MANIFEST_HASH_PATH, "w") as f:
        f.write(current_hash)


# 🔹 Function to Retrieve Metadata from FAISS
def get_dbt_metadata(query, top_k=3):
    """Retrieve the most relevant dbt metadata based on a natural language query."""
    # Ensure metadata is up to date
    update_dbt_metadata()

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Encode query into an embedding
    query_embedding = model.encode([query])

    # Search FAISS for top_k closest matches
    _, indices = index.search(np.array(query_embedding), k=top_k)

    # Load metadata
    with open(DBT_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Retrieve matching metadata
    matched_metadata = [metadata[idx] for idx in indices[0] if idx < len(metadata)]

    # Format response
    formatted_metadata = "\n\n".join([
        f"Model: {item['model_name']}\nDescription: {item['description']}\nColumns: {', '.join(item['columns'].keys())}"
        for item in matched_metadata
    ])
    
    return formatted_metadata if formatted_metadata else "No relevant metadata found."


# Ensure FAISS is up to date when this script runs
update_dbt_metadata()

