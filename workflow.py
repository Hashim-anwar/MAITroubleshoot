from __future__ import annotations

import hashlib
import io
import json
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
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

try:
    import faiss
except ImportError:
    faiss = None

from prompts import SYSTEM_PROMPT, TROUBLESHOOTING_PROMPT, QUERY_REWRITE_PROMPT

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
MAX_RETRIEVAL = 8
MAX_WEB_RESULTS = 5
# ============================================================
# BACKEND GOOGLE DRIVE OEM MANUAL FOLDER
# ============================================================

BACKEND_GOOGLE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/u/0/folders/1J0hTZ--m2cVfFUY4F4C3Dw4XN5uVsGZe"
)

def get_api_status() -> dict[str, bool]:
    return {
        "groq": bool(os.getenv("GROQ_API_KEY") or _streamlit_secret("GROQ_API_KEY")),
        "tavily": bool(os.getenv("TAVILY_API_KEY") or _streamlit_secret("TAVILY_API_KEY")),
    }


def _streamlit_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def _get_secret(name: str) -> str | None:
    return os.getenv(name) or _streamlit_secret(name)


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _new_kb() -> dict[str, Any]:
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed. Check requirements.txt.")
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
        return {"documents": 0, "chunks": 0}
    return {
        "documents": len(kb.get("documents", {})),
        "chunks": len(kb.get("chunks", [])),
    }


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            split_at = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )
            if split_at > start + MAX_CHUNK_CHARS // 2:
                end = split_at + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return [c for c in chunks if len(c) >= 40]


def _extract_pdf(data: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page_no, page in enumerate(reader.pages, 1):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append({"text": text, "page": page_no})
    return pages


def _extract_docx(data: bytes) -> list[dict[str, Any]]:
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": _clean_text(text), "page": None}] if text.strip() else []


def _extract_text(data: bytes) -> list[dict[str, Any]]:
    text = data.decode("utf-8", errors="replace")
    return [{"text": _clean_text(text), "page": None}] if text.strip() else []


def extract_document(filename: str, data: bytes) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    if suffix in {".txt", ".md"}:
        return _extract_text(data)
    raise ValueError(f"Unsupported file type: {suffix}")


def _document_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ensure_kb(kb: dict[str, Any] | None) -> dict[str, Any]:
    if kb is None:
        return _new_kb()
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed.")
    return kb


def _add_chunks(
    kb: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if not chunks:
        return kb

    texts = [x["text"] for x in chunks]
    vectors = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    if kb["index"] is None:
        kb["index"] = faiss.IndexFlatIP(vectors.shape[1])

    kb["index"].add(vectors)
    kb["chunks"].extend(chunks)
    kb["vectors"] += len(chunks)
    return kb


def ingest_document_bytes(
    filename: str,
    data: bytes,
    kb: dict[str, Any] | None,
    manufacturer: str,
    engine_model: str,
) -> dict[str, Any]:
    kb = _ensure_kb(kb)
    digest = _document_hash(data)
    if digest in kb["documents"]:
        return kb

    pages = extract_document(filename, data)
    if not pages:
        raise ValueError(f"No extractable text found in {filename}. Scanned PDFs may require OCR.")

    all_chunks = []
    for page in pages:
        for chunk_id, text in enumerate(_chunk_text(page["text"])):
            all_chunks.append(
                {
                    "text": text,
                    "file_name": filename,
                    "page": page.get("page"),
                    "manufacturer": manufacturer,
                    "engine_model": engine_model,
                    "doc_type": Path(filename).suffix.lower().lstrip("."),
                    "chunk_id": chunk_id,
                    "document_hash": digest,
                    "source_type": "Uploaded manual",
                    "title": filename,
                }
            )

    if not all_chunks:
        raise ValueError(f"No useful chunks were created from {filename}.")

    _add_chunks(kb, all_chunks)
    kb["documents"][digest] = {
        "file_name": filename,
        "manufacturer": manufacturer,
        "engine_model": engine_model,
        "chunks": len(all_chunks),
    }
    return kb


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
            raise ValueError(f"{file.name} is empty.")
        kb = ingest_document_bytes(
            file.name,
            data,
            kb,
            manufacturer,
            engine_model,
        )
    return kb

# ============================================================
# BACKEND GOOGLE DRIVE FOLDER FUNCTIONS
# ============================================================

def _extract_drive_folder_id(url: str) -> str:
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    raise ValueError(
        "Invalid Google Drive folder link. "
        "Use a link like https://drive.google.com/drive/folders/FOLDER_ID"
    )


def ingest_backend_google_drive(
    kb: dict[str, Any] | None,
) -> dict[str, Any]:

    if not BACKEND_GOOGLE_DRIVE_FOLDER_URL.strip():
        return _ensure_kb(kb)

    kb = _ensure_kb(kb)

    folder_id = _extract_drive_folder_id(
        BACKEND_GOOGLE_DRIVE_FOLDER_URL
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="marine_oem_"))

    try:
        downloaded_path = gdown.download_folder(
            id=folder_id,
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
        )

        if not downloaded_path:
            raise RuntimeError(
                "Google Drive folder could not be downloaded. "
                "Check that the folder is shared as 'Anyone with the link - Viewer'."
            )

        allowed_extensions = {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
        }

        files_found = [
            p for p in temp_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in allowed_extensions
        ]

        if not files_found:
            raise RuntimeError(
                "No supported manuals were found in the Google Drive folder."
            )

        for file_path in files_found:

            try:
                data = file_path.read_bytes()

                if not data:
                    continue

                # Use filename as initial identification.
                filename = file_path.name

                manufacturer = "Other"
                engine_model = "Backend OEM Manual"

                filename_upper = filename.upper()

                # Manufacturer detection
                if "MTU" in filename_upper:
                    manufacturer = "MTU"
                elif "CATERPILLAR" in filename_upper or "CAT " in filename_upper:
                    manufacturer = "Caterpillar"
                elif "MAN" in filename_upper:
                    manufacturer = "MAN"
                elif "YANMAR" in filename_upper:
                    manufacturer = "Yanmar"
                elif "YAMAHA" in filename_upper:
                    manufacturer = "Yamaha"
                elif "VOLVO" in filename_upper:
                    manufacturer = "Volvo Penta"

                # Engine model detection from filename
                for model in SUPPORTED_MODELS:
                    if model.upper() in filename_upper:
                        engine_model = model
                        break

                kb = ingest_document_bytes(
                    filename=filename,
                    data=data,
                    kb=kb,
                    manufacturer=manufacturer,
                    engine_model=engine_model,
                )

            except Exception as exc:
                print(
                    f"Backend manual skipped: "
                    f"{file_path.name} -> {exc}"
                )

        return kb

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
def _drive_download_url(url: str) -> str:
    match = re.search(r"/file/d/([^/]+)", url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    match = re.search(r"[?&]id=([^&]+)", url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    raise ValueError("Unsupported Google Drive link. Use a share link containing a file ID.")


def ingest_google_drive_link(
    url: str,
    kb: dict[str, Any] | None,
    manufacturer: str,
    engine_model: str,
) -> dict[str, Any]:
    direct_url = _drive_download_url(url.strip())
    response = requests.get(
        direct_url,
        timeout=45,
        headers={"User-Agent": "Marine-AI-Troubleshooting-Agent/1.0"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()

    # Google Drive can return an HTML confirmation page for large files.
    if "text/html" in content_type and not response.content.startswith(b"%PDF"):
        raise ValueError(
            "Google Drive returned an HTML confirmation/login page. "
            "Make the file accessible by link and use a direct downloadable file."
        )

    filename = "google_drive_manual.pdf" if response.content.startswith(b"%PDF") else "google_drive_manual"
    if "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type:
        filename += ".docx"
    elif "text/plain" in content_type:
        filename += ".txt"
    else:
        filename += ".pdf"

    return ingest_document_bytes(
        filename,
        response.content,
        kb,
        manufacturer,
        engine_model,
    )


def retrieve(
    kb: dict[str, Any] | None,
    query: str,
    engine_model: str,
    manufacturer: str,
    serial_number: str = "",
    top_k: int = MAX_RETRIEVAL,
) -> list[dict[str, Any]]:
    if not kb or kb.get("index") is None or not kb.get("chunks"):
        return []

    q_vector = get_embedder().encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    scores, indices = kb["index"].search(q_vector, min(top_k * 4, len(kb["chunks"])))

    candidates = []
    model_norm = engine_model.lower().strip()
    manufacturer_norm = manufacturer.lower().strip()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(kb["chunks"][int(idx)])
        text = item["text"].lower()
        model_match = model_norm and model_norm in item.get("engine_model", "").lower()
        manu_match = manufacturer_norm and manufacturer_norm in item.get("manufacturer", "").lower()
        exact_model_text = model_norm and model_norm in text
        bonus = 0.0
        if model_match:
            bonus += 0.30
        if manu_match:
            bonus += 0.10
        if exact_model_text:
            bonus += 0.15
        item["score"] = float(score) + bonus
        candidates.append(item)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def _tavily_search(query: str) -> list[dict[str, Any]]:
    api_key = _get_secret("TAVILY_API_KEY")
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
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "source_type": "Online research",
        }
        for item in data.get("results", [])
    ]


def _format_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "No uploaded-manual evidence was retrieved."
    blocks = []
    for i, item in enumerate(evidence, 1):
        blocks.append(
            f"[MANUAL EVIDENCE {i}]\n"
            f"File: {item.get('file_name', item.get('title', 'Unknown'))}\n"
            f"Page: {item.get('page', 'N/A')}\n"
            f"Manufacturer: {item.get('manufacturer', 'N/A')}\n"
            f"Engine model: {item.get('engine_model', 'N/A')}\n"
            f"Source type: {item.get('source_type', 'N/A')}\n"
            f"Text:\n{item.get('text', '')}"
        )
    return "\n\n".join(blocks)


def _format_web(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No online research results were retrieved."
    blocks = []
    for i, item in enumerate(results, 1):
        blocks.append(
            f"[WEB RESULT {i}]\n"
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content:\n{item.get('content', '')}"
        )
    return "\n\n".join(blocks)


def _groq_client() -> Groq:
    key = _get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets or your environment."
        )
    return Groq(api_key=key)


def _technical_query(context: dict[str, str], defect: str) -> str:
    return (
        f"{context['manufacturer']} {context['engine_model']} "
        f"serial {context.get('serial_number','')} "
        f"{defect}"
    )


def troubleshoot(
    context: dict[str, str],
    defect: str,
    kb: dict[str, Any] | None,
    search_mode: str = "Uploaded Manuals",
) -> dict[str, Any]:
    query = _technical_query(context, defect)

    evidence = retrieve(
        kb,
        query,
        engine_model=context["engine_model"],
        manufacturer=context["manufacturer"],
        serial_number=context.get("serial_number", ""),
    )

    web_results: list[dict[str, Any]] = []
    web_used = False
    if search_mode in {"Online Research", "Both"}:
        if _get_secret("TAVILY_API_KEY"):
            try:
                web_results = _tavily_search(query)
                web_used = bool(web_results)
            except Exception as exc:
                web_results = [
                    {
                        "title": "Online research error",
                        "url": "",
                        "content": str(exc),
                        "source_type": "Online research",
                    }
                ]

    evidence_text = _format_evidence(evidence)
    web_text = _format_web(web_results)

    prompt = TROUBLESHOOTING_PROMPT.format(
        manufacturer=context["manufacturer"],
        engine_model=context["engine_model"],
        serial_number=context.get("serial_number") or "Not provided",
        vessel_name=context.get("vessel_name") or "Not provided",
        operating_hours=context.get("operating_hours") or "Not provided",
        application_type=context.get("application_type") or "Not provided",
        defect=defect,
        evidence=evidence_text,
        web_results=web_text,
    )

    client = _groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=5000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response.choices[0].message.content or "No response was generated."

    return {
        "answer": answer,
        "evidence": evidence,
        "web_results": web_results,
        "web_used": web_used,
        "query": query,
    }
