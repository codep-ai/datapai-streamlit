# agents/knowledge_ingest_agent.py

from __future__ import annotations

import os
import mimetypes
from typing import List, Optional
from urllib.parse import urlparse

import boto3
import streamlit as st

from embeddings.embed import embed_texts  # your existing embedding function

boto3.setup_default_session(region_name="ap-southeast-2")
# -------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------

def load_text_file(path: str) -> str:
    """
    Load a plain text file with a few fallback encodings.
    Used for .txt/.md/.csv when needed.
    """
    for encoding in ["utf-8", "utf-16", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            st.error(f"File not found: {path}")
            return ""
    st.error(f"Unable to decode file {path} with common encodings.")
    return ""


def extract_pdf_text(path: str) -> str:
    """
    Extract text from a PDF for semantic search.
    Uses pdfplumber (cheap & local).
    """
    try:
        import pdfplumber
    except ImportError:
        st.error("pdfplumber is not installed. Run: pip install pdfplumber")
        return ""

    text_chunks: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text)
    except Exception as e:
        st.error(f"Failed to read PDF {path}: {e}")
        return ""

    return "\n\n".join(text_chunks)


def extract_pdf_text_textract(s3_bucket: str, s3_key: str) -> str:
    """
    Use AWS Textract to extract text from a PDF stored in S3.

    NOTE:
    - This uses synchronous DetectDocumentText API.
    - Good for small/medium documents.
    - For very large PDFs, consider async APIs.
    """
    textract = boto3.client("textract")
    text_lines: List[str] = []

    try:
        response = textract.detect_document_text(
            Document={
                "S3Object": {
                    "Bucket": s3_bucket,
                    "Name": s3_key,
                }
            }
        )
        for block in response.get("Blocks", []):
            if block.get("BlockType") == "LINE":
                line = block.get("Text", "").strip()
                if line:
                    text_lines.append(line)
    except Exception as e:
        st.error(f"Textract failed for s3://{s3_bucket}/{s3_key}: {e}")
        return ""

    return "\n".join(text_lines)


def upload_to_s3(file_path: str, bucket_name: str, s3_key: str) -> str:
    """
    Upload a local file to S3 and return its S3 URI.
    """
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        st.error(f"Error uploading {file_path} to S3: {e}")
        raise


# -------------------------------------------------------------------
# Record builders for LanceDB
# -------------------------------------------------------------------

def process_pdf_file(file_path: str, s3_bucket: str) -> Optional[dict]:
    """
    Extract text from a PDF, upload original to S3, embed the text,
    and return a LanceDB record.

    Flow:
    1. Upload PDF to S3.
    2. Try local text extraction with pdfplumber.
    3. If empty, fallback to AWS Textract OCR.
    4. If still empty, store a minimal placeholder so the file is indexed.
    """
    s3_key = f"raw_data/{os.path.basename(file_path)}"
    s3_uri = upload_to_s3(file_path, s3_bucket, s3_key)

    # 1. Try local extraction
    text = extract_pdf_text(file_path)

    # 2. If nothing, fallback to Textract
    if not text.strip():
        st.warning(f"No text extracted with pdfplumber for {file_path}, trying Textract OCR...")
        text = extract_pdf_text_textract(s3_bucket, s3_key)

    # 3. If still nothing, store minimal info but don't drop the file
    if not text.strip():
        st.warning(f"No text extracted from PDF even with Textract: {file_path}")
        text = (
            f"PDF file {os.path.basename(file_path)} stored at {s3_uri}. "
            f"No readable text extracted; likely a purely image-based or unsupported document."
        )

    # 4. Embed whatever text we have
    vector = embed_texts([text])[0]

    return {
        "vector": vector,
        "text": text,
        "source_uri": s3_uri,
        "filename": os.path.basename(file_path),
        "collection": "pdfs",
    }


def process_image_file(file_path: str, s3_bucket: str) -> Optional[dict]:
    """
    Upload image to S3, create a simple caption string, embed that,
    and return a LanceDB record.

    NOTE: This version does NOT use a vision model (no extra disk space).
    It creates a generic caption that still gives you a searchable text field.
    """
    s3_key = f"raw_data/{os.path.basename(file_path)}"
    s3_uri = upload_to_s3(file_path, s3_bucket, s3_key)

    # Simple caption; later you can replace this with a real vision model.
    caption = f"Image file {os.path.basename(file_path)} stored at {s3_uri}"

    vector = embed_texts([caption])[0]

    return {
        "vector": vector,
        "text": caption,
        "source_uri": s3_uri,
        "filename": os.path.basename(file_path),
        "collection": "images",
    }


def process_text_file(file_path: str) -> Optional[dict]:
    """
    Process plain text / md / csv files.
    Not your primary business value, but kept for completeness.
    """
    content = load_text_file(file_path)
    if not content.strip():
        st.warning(f"Empty text content in file: {file_path}")
        return None

    vector = embed_texts([content])[0]

    return {
        "vector": vector,
        "text": content,
        "source_uri": f"file://{os.path.abspath(file_path)}",
        "filename": os.path.basename(file_path),
        "collection": "documents",
    }


def process_multimodal_file(file_path: str, s3_bucket: str) -> Optional[dict]:
    """
    Route PDF/image files to the proper processing function.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return process_pdf_file(file_path, s3_bucket)

    if ext in [".png", ".jpg", ".jpeg"]:
        return process_image_file(file_path, s3_bucket)

    st.warning(f"Unsupported multimodal file type for {file_path}")
    return None


# -------------------------------------------------------------------
# LanceDB ingestion
# -------------------------------------------------------------------

def ingest_files_to_lancedb(file_paths: List[str], db_uri: str = "vector_store"):
    """
    Ingest a list of files into LanceDB.

    - If db_uri is s3://bucket/path, we:
      * connect LanceDB to that S3 URI
      * upload raw PDFs/images to the same bucket under raw_data/
    - If db_uri is local, we still ingest, but won't upload to S3 unless configured.
    """
    import lancedb

    raw_bucket_name: Optional[str] = None

    if db_uri.startswith("s3://"):
        parsed_uri = urlparse(db_uri)
        raw_bucket_name = parsed_uri.netloc  # reuse the LanceDB bucket for raw_data
        st.info(f"Connecting to LanceDB on S3 URI: {db_uri}")
    else:
        st.info(f"Connecting to local LanceDB path: {db_uri}")

    # Optionally override raw data bucket (if you want a separate one)
    raw_bucket_name = os.environ.get("RAW_DATA_BUCKET", raw_bucket_name)

    db = lancedb.connect(db_uri)

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        mime_type, _ = mimetypes.guess_type(file_path)

        data_record: Optional[dict] = None

        # PDF or image: require an S3 bucket to store raw_data
        if mime_type and ("image" in mime_type or "pdf" in mime_type):
            if not raw_bucket_name:
                st.warning(
                    f"Cannot process {file_path} as multimodal without an S3-backed "
                    f"LanceDB URI or RAW_DATA_BUCKET env."
                )
                continue
            data_record = process_multimodal_file(file_path, raw_bucket_name)

        # Text-like files
        elif (mime_type and "text" in mime_type) or ext in [".txt", ".md", ".csv"]:
            data_record = process_text_file(file_path)

        else:
            st.warning(f"Skipping unsupported file type: {file_path}")
            continue

        if not data_record:
            continue

        collection = data_record.pop("collection")

        # Upsert into LanceDB
        if collection in db.table_names():
            table = db.open_table(collection)
            table.add([data_record])
        else:
            db.create_table(collection, data=[data_record], mode="overwrite")

        st.success(f"Ingested into collection '{collection}': {file_path}")


# -------------------------------------------------------------------
# Public entry point used by Streamlit tab
# -------------------------------------------------------------------

def run_knowledge_ingest_agent(file_paths: List[str]):
    """
    Streamlit-friendly entry point.

    Uses LANCEDB_URI env if set, otherwise defaults to your original S3 bucket.
    This keeps backward compatibility with your existing Streamlit tab.
    """
    st.write("📥 Starting knowledge ingestion...")

    default_uri = "s3://codepais3/lancedb_data/"
    db_uri = os.environ.get("LANCEDB_URI", default_uri)

    with st.spinner(f"Ingesting documents into LanceDB at {db_uri} ..."):
        ingest_files_to_lancedb(file_paths, db_uri=db_uri)

    st.success("Knowledge ingestion completed!")

