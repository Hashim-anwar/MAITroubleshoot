```python
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import requests
import streamlit as st
import gdown
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    faiss = None

from prompts import (
    SYSTEM_PROMPT,
    TROUBLESHOOTING_PROMPT,
)


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_MANUFACTURERS = [
    "MTU",
    "Caterpillar",
    "MAN",
    "Yamaha",
    "Yanmar",
    "Volvo Penta",
    "Other",
]

SUPPORTED_MODELS = [
    "MTU 10V 2000 M94",
    "MTU 12V 2000 M94",
    "MTU 12V 2000 M96L",
    "MTU 16V 4000 M90",
    "MAN 12V 175D",
    "MAN 16V 175D",
]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP = 250

# Number of manual chunks sent to the AI
MAX_RETRIEVAL = 6

# Number of online results
MAX_WEB_RESULTS = 4

# Maximum characters from each retrieved manual chunk
MAX_EVIDENCE_CHARS = 3500

# Maximum characters from each web result
MAX_WEB_CHARS = 2500


# ============================================================
# BACKEND GOOGLE DRIVE OEM MANUAL FOLDER
# ============================================================
#
# PUT YOUR GOOGLE DRIVE FOLDER LINK HERE.
#
# Example:
#
# BACKEND_GOOGLE_DRIVE_FOLDER_URL = (
#     "https://drive.google.com/drive/folders/1J0hTZ--m2cVfFUY4F4C3Dw4XN5uVsGZe?usp=drive_link"
# )
#
# The folder should be shared:
# Anyone with the link -> Viewer
#
# ============================================================

BACKEND_GOOGLE_DRIVE_FOLDER_URL = (
    "PASTE YOUR GOOGLE DRIVE FOLDER LINK HERE"
)


# ============================================================
# API / SECRET HELPERS
# ============================================================

def _streamlit_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    return None


def _get_secret(name: str) -> str | None:
    return os.getenv(name) or _streamlit_secret(name)


def get_api_status() -> dict[str, bool]:
    return {
        "groq": bool(_get_secret("GROQ_API_KEY")),
        "tavily": bool(_get_secret("TAVILY_API_KEY")),
    }


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

def _new_kb() -> dict[str, Any]:
    if faiss is None:
        raise RuntimeError(
            "faiss-cpu is not installed. Check requirements.txt."
        )

    return {
        "index": None,
        "vectors": 0,
        "chunks": [],
        "documents": {},
    }


def clear_knowledge_base() -> dict[str, Any]:
    return _new_kb()


def get_kb_stats(kb: dict[str, Any] | None) -> dict[str, int]:
    if not kb:
        return {
            "documents": 0,
            "chunks": 0,
        }

    return {
        "documents": len(kb.get("documents", {})),
        "chunks": len(kb.get("chunks", [])),
    }


def _ensure_kb(kb: dict[str, Any] | None) -> dict[str, Any]:
    if kb is None:
        return _new_kb()

    if faiss is None:
        raise RuntimeError(
            "faiss-cpu is not installed. Check requirements.txt."
        )

    return kb


# ============================================================
# TEXT CLEANING / CHUNKING
# ============================================================

def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")

    # Normalize spaces but preserve line breaks.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _chunk_text(text: str) -> list[str]:
    text = _clean_text(text)

    if not text:
        return []

    chunks: list[str] = []

    start = 0

    while start < len(text):
        end = min(
            start + MAX_CHUNK_CHARS,
            len(text),
        )

        if end < len(text):
            split_at = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )

            if split_at > start + MAX_CHUNK_CHARS // 2:
                end = split_at + 1

        chunk = text[start:end].strip()

        if len(chunk) >= 40:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            0,
            end - CHUNK_OVERLAP,
        )

    return chunks


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def _extract_pdf(data: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(data))

    pages: list[dict[str, Any]] = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = _clean_text(
            page.extract_text() or ""
        )

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_no,
                }
            )

    return pages


def _extract_docx(data: bytes) -> list[dict[str, Any]]:
    document = Document(io.BytesIO(data))

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    text = "\n".join(paragraphs)

    text = _clean_text(text)

    if not text:
        return []

    return [
        {
            "text": text,
            "page": None,
        }
    ]


def _extract_text(data: bytes) -> list[dict[str, Any]]:
    text = data.decode(
        "utf-8",
        errors="replace",
    )

    text = _clean_text(text)

    if not text:
        return []

    return [
        {
            "text": text,
            "page": None,
        }
    ]


def extract_document(
    filename: str,
    data: bytes,
) -> list[dict[str, Any]]:

    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(data)

    if suffix == ".docx":
        return _extract_docx(data)

    if suffix in {".txt", ".md"}:
        return _extract_text(data)

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )


# ============================================================
# DOCUMENT HASH
# ============================================================

def _document_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# ENGINE / MANUFACTURER DETECTION
# ============================================================

def _normalize_identifier(text: str) -> str:
    text = text.upper()

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def detect_manufacturer(
    filename: str,
    text: str = "",
) -> str:

    combined = _normalize_identifier(
        f"{filename} {text[:15000]}"
    )

    # MTU
    if "MTU" in combined:
        return "MTU"

    # Caterpillar / CAT
    if (
        "CATERPILLAR" in combined
        or re.search(r"\bCAT\s*(C12|C18|C32|3508|3512|3516)\b", combined)
    ):
        return "Caterpillar"

    # MAN
    if re.search(
        r"\bMAN\b|\b175D\b",
        combined,
    ):
        return "MAN"

    # Yamaha
    if "YAMAHA" in combined:
        return "Yamaha"

    # Yanmar
    if "YANMAR" in combined:
        return "Yanmar"

    # Volvo Penta
    if (
        "VOLVO PENTA" in combined
        or "VOLVO" in combined
    ):
        return "Volvo Penta"

    return "Other"


def detect_engine_model(
    filename: str,
    text: str = "",
) -> str:

    combined = _normalize_identifier(
        f"{filename} {text[:20000]}"
    )

    # --------------------------------------------------------
    # Exact supported models
    # --------------------------------------------------------

    model_patterns = [
        (
            "MTU 10V 2000 M94",
            [
                r"\b10V\s*2000\s*M94\b",
                r"\b10V\s*2000\s*M-94\b",
            ],
        ),
        (
            "MTU 12V 2000 M94",
            [
                r"\b12V\s*2000\s*M94\b",
                r"\b12V\s*2000\s*M-94\b",
            ],
        ),
        (
            "MTU 12V 2000 M96L",
            [
                r"\b12V\s*2000\s*M96L\b",
                r"\b12V\s*2000\s*M-96L\b",
            ],
        ),
        (
            "MTU 16V 4000 M90",
            [
                r"\b16V\s*4000\s*M90\b",
                r"\b16V\s*4000\s*M-90\b",
            ],
        ),
        (
            "MAN 12V 175D",
            [
                r"\b12V\s*175D\b",
                r"\b12V\s*175D[- ]?ML\b",
            ],
        ),
        (
            "MAN 16V 175D",
            [
                r"\b16V\s*175D\b",
                r"\b16V\s*175D[- ]?MM\b",
            ],
        ),
    ]

    for model_name, patterns in model_patterns:
        for pattern in patterns:
            if re.search(pattern, combined):
                return model_name

    return "Other / Model Not Detected"


def detect_document_identity(
    filename: str,
    text: str,
) -> tuple[str, str]:

    manufacturer = detect_manufacturer(
        filename,
        text,
    )

    engine_model = detect_engine_model(
        filename,
        text,
    )

    return manufacturer, engine_model


# ============================================================
# ADD EMBEDDINGS / CHUNKS
# ============================================================

def _add_chunks(
    kb: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:

    if not chunks:
        return kb

    texts = [
        item["text"]
        for item in chunks
    ]

    vectors = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    if kb["index"] is None:
        kb["index"] = faiss.IndexFlatIP(
            vectors.shape[1]
        )

    kb["index"].add(vectors)

    kb["chunks"].extend(chunks)

    kb["vectors"] += len(chunks)

    return kb


# ============================================================
# INGEST ONE DOCUMENT
# ============================================================

def ingest_document_bytes(
    filename: str,
    data: bytes,
    kb: dict[str, Any] | None,
    manufacturer: str = "Other",
    engine_model: str = "Other / Model Not Detected",
    source_type: str = "Uploaded manual",
) -> dict[str, Any]:

    kb = _ensure_kb(kb)

    digest = _document_hash(data)

    # Avoid indexing the same document twice.
    if digest in kb["documents"]:
        return kb

    pages = extract_document(
        filename,
        data,
    )

    if not pages:
        raise ValueError(
            f"No extractable text found in {filename}. "
            "Scanned PDFs may require OCR."
        )

    # Combine enough text for automatic identity detection.
    identity_text = "\n".join(
        page["text"]
        for page in pages
    )[:30000]

    detected_manufacturer, detected_model = (
        detect_document_identity(
            filename,
            identity_text,
        )
    )

    # Automatic detection takes priority if successful.
    if detected_manufacturer != "Other":
        manufacturer = detected_manufacturer

    if detected_model != "Other / Model Not Detected":
        engine_model = detected_model

    all_chunks: list[dict[str, Any]] = []

    for page in pages:

        page_chunks = _chunk_text(
            page["text"]
        )

        for chunk_id, text in enumerate(
            page_chunks
        ):

            all_chunks.append(
                {
                    "text": text,
                    "file_name": filename,
                    "page": page.get("page"),
                    "manufacturer": manufacturer,
                    "engine_model": engine_model,
                    "doc_type": (
                        Path(filename)
                        .suffix
                        .lower()
                        .lstrip(".")
                    ),
                    "chunk_id": chunk_id,
                    "document_hash": digest,
                    "source_type": source_type,
                    "title": filename,
                }
            )

    if not all_chunks:
        raise ValueError(
            f"No useful chunks were created from {filename}."
        )

    _add_chunks(
        kb,
        all_chunks,
    )

    kb["documents"][digest] = {
        "file_name": filename,
        "manufacturer": manufacturer,
        "engine_model": engine_model,
        "chunks": len(all_chunks),
        "source_type": source_type,
    }

    return kb


# ============================================================
# MANUAL FILE UPLOAD
# ============================================================

def ingest_uploaded_files(
    files: list[BinaryIO],
    kb: dict[str, Any] | None,
    manufacturer: str,
    engine_model: str,
) -> dict[str, Any]:

    kb = _ensure_kb(kb)

    for file in files:

        data = file.getvalue()

        if not data:
            raise ValueError(
                f"{file.name} is empty."
            )

        kb = ingest_document_bytes(
            filename=file.name,
            data=data,
            kb=kb,
            manufacturer=manufacturer,
            engine_model=engine_model,
            source_type="Uploaded manual",
        )

    return kb


# ============================================================
# GOOGLE DRIVE HELPERS
# ============================================================

def _extract_drive_folder_id(
    url: str,
) -> str:

    match = re.search(
        r"/folders/([a-zA-Z0-9_-]+)",
        url,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"[?&]id=([a-zA-Z0-9_-]+)",
        url,
    )

    if match:
        return match.group(1)

    raise ValueError(
        "Invalid Google Drive folder link. "
        "Use a link like: "
        "https://drive.google.com/drive/folders/FOLDER_ID"
    )


def _extract_drive_file_id(
    url: str,
) -> str:

    match = re.search(
        r"/file/d/([^/]+)",
        url,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"[?&]id=([^&]+)",
        url,
    )

    if match:
        return match.group(1)

    raise ValueError(
        "Unsupported Google Drive file link."
    )


def _drive_download_url(
    url: str,
) -> str:

    file_id = _extract_drive_file_id(
        url.strip()
    )

    return (
        "https://drive.google.com/"
        f"uc?export=download&id={file_id}"
    )


# ============================================================
# BACKEND GOOGLE DRIVE FOLDER INGESTION
# ============================================================

def ingest_backend_google_drive(
    kb: dict[str, Any] | None,
) -> dict[str, Any]:

    kb = _ensure_kb(kb)

    folder_url = (
        BACKEND_GOOGLE_DRIVE_FOLDER_URL.strip()
    )

    if not folder_url:
        raise RuntimeError(
            "BACKEND_GOOGLE_DRIVE_FOLDER_URL is empty. "
            "Add your Google Drive folder link in workflow.py."
        )

    if "PASTE YOUR GOOGLE DRIVE" in folder_url:
        raise RuntimeError(
            "You have not added the Google Drive folder link yet."
        )

    folder_id = _extract_drive_folder_id(
        folder_url
    )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="marine_oem_"
        )
    )

    try:

        downloaded_path = gdown.download_folder(
            id=folder_id,
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
            remaining_ok=True,
        )

        if not downloaded_path:
            raise RuntimeError(
                "Google Drive folder could not be downloaded. "
                "Check that the folder is shared as "
                "'Anyone with the link - Viewer'."
            )

        allowed_extensions = {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
        }

        files_found = [
            path
            for path in temp_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in allowed_extensions
            )
        ]

        if not files_found:
            raise RuntimeError(
                "No PDF, DOCX, TXT or MD manuals were found "
                "in the Google Drive folder."
            )

        indexed_count = 0

        for file_path in files_found:

            try:

                data = file_path.read_bytes()

                if not data:
                    continue

                filename = file_path.name

                kb = ingest_document_bytes(
                    filename=filename,
                    data=data,
                    kb=kb,
                    manufacturer="Other",
                    engine_model=(
                        "Other / Model Not Detected"
                    ),
                    source_type=(
                        "Backend Google Drive OEM manual"
                    ),
                )

                indexed_count += 1

            except Exception as exc:

                # Do not stop the entire ingestion process
                # because one manual has a problem.
                print(
                    f"Backend manual skipped: "
                    f"{file_path.name} -> {exc}"
                )

        if indexed_count == 0:
            raise RuntimeError(
                "No backend manuals could be indexed."
            )

        return kb

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ============================================================
# SINGLE GOOGLE DRIVE FILE INGESTION
# ============================================================

def ingest_google_drive_link(
    url: str,
    kb: dict[str, Any] | None,
    manufacturer: str,
    engine_model: str,
) -> dict[str, Any]:

    direct_url = _drive_download_url(
        url.strip()
    )

    response = requests.get(
        direct_url,
        timeout=60,
        headers={
            "User-Agent": (
                "Marine-AI-Troubleshooting-Agent/1.0"
            )
        },
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    content = response.content

    # Google Drive can return an HTML page instead
    # of the actual document.
    if (
        "text/html" in content_type
        and not content.startswith(b"%PDF")
    ):
        raise ValueError(
            "Google Drive returned an HTML page instead "
            "of the document. Make sure the file is shared "
            "as 'Anyone with the link - Viewer'."
        )

    if content.startswith(b"%PDF"):
        filename = "google_drive_manual.pdf"

    elif (
        "wordprocessingml.document"
        in content_type
    ):
        filename = "google_drive_manual.docx"

    elif "text/plain" in content_type:
        filename = "google_drive_manual.txt"

    else:
        filename = "google_drive_manual.pdf"

    return ingest_document_bytes(
        filename=filename,
        data=content,
        kb=kb,
        manufacturer=manufacturer,
        engine_model=engine_model,
        source_type="Google Drive manual",
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    kb: dict[str, Any] | None,
    query: str,
    engine_model: str,
    manufacturer: str,
    serial_number: str = "",
    top_k: int = MAX_RETRIEVAL,
) -> list[dict[str, Any]]:

    if (
        not kb
        or kb.get("index") is None
        or not kb.get("chunks")
    ):
        return []

    query_vector = get_embedder().encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    search_count = min(
        top_k * 5,
        len(kb["chunks"]),
    )

    scores, indices = kb["index"].search(
        query_vector,
        search_count,
    )

    model_norm = _normalize_identifier(
        engine_model
    )

    manufacturer_norm = _normalize_identifier(
        manufacturer
    )

    candidates: list[dict[str, Any]] = []

    for score, index in zip(
        scores[0],
        indices[0],
    ):

        if index < 0:
            continue

        item = dict(
            kb["chunks"][int(index)]
        )

        item_model = _normalize_identifier(
            item.get(
                "engine_model",
                "",
            )
        )

        item_manufacturer = _normalize_identifier(
            item.get(
                "manufacturer",
                "",
            )
        )

        text_norm = _normalize_identifier(
            item.get(
                "text",
                "",
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Exact model gets a strong preference.
        # ----------------------------------------------------

        exact_model_match = (
            model_norm
            and model_norm != "OTHER MODEL NOT DETECTED"
            and model_norm in item_model
        )

        manufacturer_match = (
            manufacturer_norm
            and manufacturer_norm in item_manufacturer
        )

        model_text_match = (
            model_norm
            and model_norm in text_norm
        )

        # Penalize clearly different known models.
        different_model_penalty = 0.0

        item_known_model = (
            item_model
            and item_model != "OTHER MODEL NOT DETECTED"
        )

        if (
            model_norm
            and item_known_model
            and model_norm != item_model
        ):
            different_model_penalty = 0.35

        bonus = 0.0

        if exact_model_match:
            bonus += 0.40

        if manufacturer_match:
            bonus += 0.12

        if model_text_match:
            bonus += 0.18

        final_score = (
            float(score)
            + bonus
            - different_model_penalty
        )

        item["score"] = final_score

        candidates.append(item)

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates[:top_k]


# ============================================================
# TAVILY ONLINE RESEARCH
# ============================================================

def _tavily_search(
    query: str,
) -> list[dict[str, Any]]:

    api_key = _get_secret(
        "TAVILY_API_KEY"
    )

    if not api_key:
        return []

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": MAX_WEB_RESULTS,
            "include_answer": False,
        },
        timeout=45,
    )

    response.raise_for_status()

    data = response.json()

    return [
        {
            "title": item.get(
                "title",
                "",
            ),
            "url": item.get(
                "url",
                "",
            ),
            "content": item.get(
                "content",
                "",
            ),
            "source_type": "Online research",
        }
        for item in data.get(
            "results",
            [],
        )
    ]


# ============================================================
# FORMAT EVIDENCE FOR AI
# ============================================================

def _format_evidence(
    evidence: list[dict[str, Any]],
) -> str:

    if not evidence:
        return (
            "No relevant OEM manual evidence "
            "was retrieved."
        )

    blocks = []

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        text = item.get(
            "text",
            "",
        )[:MAX_EVIDENCE_CHARS]

        blocks.append(
            f"[MANUAL EVIDENCE {i}]\n"
            f"File: {item.get('file_name', item.get('title', 'Unknown'))}\n"
            f"Page: {item.get('page', 'N/A')}\n"
            f"Manufacturer: {item.get('manufacturer', 'N/A')}\n"
            f"Engine model: {item.get('engine_model', 'N/A')}\n"
            f"Source: {item.get('source_type', 'N/A')}\n"
            f"Relevance score: {item.get('score', 0):.3f}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(blocks)


def _format_web(
    results: list[dict[str, Any]],
) -> str:

    if not results:
        return (
            "No online research results "
            "were retrieved."
        )

    blocks = []

    for i, item in enumerate(
        results,
        start=1,
    ):

        content = item.get(
            "content",
            "",
        )[:MAX_WEB_CHARS]

        blocks.append(
            f"[WEB RESULT {i}]\n"
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(blocks)


# ============================================================
# GROQ
# ============================================================

def _groq_client() -> Groq:

    key = _get_secret(
        "GROQ_API_KEY"
    )

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Cloud Secrets."
        )

    return Groq(
        api_key=key
    )


# ============================================================
# TECHNICAL QUERY
# ============================================================

def _technical_query(
    context: dict[str, str],
    defect: str,
) -> str:

    parts = [
        context.get(
            "manufacturer",
            "",
        ),
        context.get(
            "engine_model",
            "",
        ),
    ]

    serial = context.get(
        "serial_number",
        "",
    )

    if serial:
        parts.append(
            f"serial {serial}"
        )

    parts.append(defect)

    return " ".join(
        part
        for part in parts
        if part
    )


# ============================================================
# MAIN TROUBLESHOOTING WORKFLOW
# ============================================================

def troubleshoot(
    context: dict[str, str],
    defect: str,
    kb: dict[str, Any] | None,
    search_mode: str = "Uploaded Manuals",
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Automatically load backend manuals if:
    #
    # 1. Google Drive backend is configured
    # 2. Current KB is empty
    #
    # This means the user does not have to manually load
    # backend manuals every time.
    # --------------------------------------------------------

    kb = _ensure_kb(kb)

    if (
        not kb.get("chunks")
        and BACKEND_GOOGLE_DRIVE_FOLDER_URL.strip()
        and "PASTE YOUR GOOGLE DRIVE"
        not in BACKEND_GOOGLE_DRIVE_FOLDER_URL
    ):

        try:
            kb = ingest_backend_google_drive(
                kb
            )

        except Exception as exc:
            # Backend loading should not prevent
            # manual-upload or online troubleshooting.
            print(
                f"Backend Google Drive loading failed: {exc}"
            )

    query = _technical_query(
        context,
        defect,
    )

    # --------------------------------------------------------
    # Retrieve OEM evidence
    # --------------------------------------------------------

    evidence = retrieve(
        kb=kb,
        query=query,
        engine_model=context[
            "engine_model"
        ],
        manufacturer=context[
            "manufacturer"
        ],
        serial_number=context.get(
            "serial_number",
            "",
        ),
        top_k=MAX_RETRIEVAL,
    )

    # --------------------------------------------------------
    # Online research
    # --------------------------------------------------------

    web_results: list[dict[str, Any]] = []

    web_used = False

    if search_mode in {
        "Online Research",
        "Both",
    }:

        if _get_secret(
            "TAVILY_API_KEY"
        ):

            try:

                web_results = _tavily_search(
                    query
                )

                web_used = bool(
                    web_results
                )

            except Exception as exc:

                web_results = [
                    {
                        "title": "Online research error",
                        "url": "",
                        "content": str(exc),
                        "source_type": (
                            "Online research"
                        ),
                    }
                ]

    # --------------------------------------------------------
    # Prepare AI context
    # --------------------------------------------------------

    evidence_text = _format_evidence(
        evidence
    )

    web_text = _format_web(
        web_results
    )

    prompt = TROUBLESHOOTING_PROMPT.format(
        manufacturer=context[
            "manufacturer"
        ],
        engine_model=context[
            "engine_model"
        ],
        serial_number=context.get(
            "serial_number"
        ) or "Not provided",
        vessel_name=context.get(
            "vessel_name"
        ) or "Not provided",
        operating_hours=context.get(
            "operating_hours"
        ) or "Not provided",
        application_type=context.get(
            "application_type"
        ) or "Not provided",
        defect=defect,
        evidence=evidence_text,
        web_results=web_text,
    )

    # --------------------------------------------------------
    # Groq AI
    # --------------------------------------------------------

    client = _groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=2600,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = (
        response.choices[0]
        .message.content
        or "No response was generated."
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "answer": answer,
        "evidence": evidence,
        "web_results": web_results,
        "web_used": web_used,
        "query": query,
        "kb": kb,
    }
```
