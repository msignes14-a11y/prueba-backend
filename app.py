# -*- coding: utf-8 -*-
import os
from io import BytesIO
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader

# ---------------------------
# Modelos (Pydantic v1)
# ---------------------------
class JurisMeta(BaseModel):
    filename: Optional[str] = None
    category: Optional[str] = None   # p.ej. "penal", "civil", etc.
    case_id: Optional[str] = None    # identificador de caso
    extra: Dict[str, Any] = Field(default_factory=dict)

class IngestItem(BaseModel):
    doc_id: str
    text: str
    meta: Optional[Union[JurisMeta, Dict[str, Any]]] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None  # p.ej. {"category":"penal"} o {"case_id":"PEN-0001"}

class QueryChunk(BaseModel):
    doc_id: str
    text: str
    score: float
    meta: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    results: List[QueryChunk]

# ---------------------------
# Utilidades
# ---------------------------
DATA_DIR = os.path.abspath("./data")
os.makedirs(DATA_DIR, exist_ok=True)

def clean_text(t: str) -> str:
    import re
    t = (t or "").replace("\x00", " ").replace("\u0000", " ")
    t = re.sub(r"[\r\t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(t: str, max_chars: int = 1500, overlap: int = 150) -> List[str]:
    """Divide texto en trozos con solapamiento."""
    t = t.strip()
    if not t:
        return []
    chunks = []
    start = 0
    n = len(t)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(t[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def meta_to_dict(meta: Optional[Union[JurisMeta, Dict[str, Any]]]) -> Dict[str, Any]:
    """Convierte meta a dict y filtra sólo tipos primitivos (str, int, float, bool)."""
    if meta is None:
        raw = {}
    elif isinstance(meta, dict):
        raw = dict(meta)
    elif isinstance(meta, JurisMeta):
        # Pydantic v1 -> .dict()
        raw = meta.dict()
    else:
        raw = dict(meta)  # mejor que explote bonito si no es soportado

    # filtra tipos válidos para Chroma
    prim: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            if v is not None:
                prim[k] = v
        # ignora listas/objetos complejos

    return prim

def make_id(doc_id: str, suffix: str) -> str:
    return f"{doc_id}__{suffix}"

# ---------------------------
# ChromaDB (modo legacy que te funciona)
# ---------------------------
# Si alguna vez ves el aviso de "deprecated configuration", es solo un warning;
# esta forma funciona bien en tu entorno actual.
client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db")
)
collection = client.get_or_create_collection("sibila")

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(
    title="SIBILA",
    description="API de SIBILA para subir (TXT/PDF), ingerir, listar, ver, borrar y consultar jurisprudencia y leyes.",
    version="1.1.0",
)

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Sube ficheros TXT o PDF a ./data (sólo guardado)."""
    saved = []
    for f in files:
        content = await f.read()
        path = os.path.join(DATA_DIR, f.filename)
        with open(path, "wb") as w:
            w.write(content)
        saved.append(f.filename)
    return {"saved": saved, "folder": DATA_DIR}

@app.post("/ingest")
def ingest(items: List[IngestItem]):
    """Ingiere documentos (texto ya pasado) y los mete troceados en Chroma."""
    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for item in items:
        chunks = chunk_text(clean_text(item.text))
        for i, chunk in enumerate(chunks):
            cid = make_id(item.doc_id, f"{i}")
            ids.append(cid)
            docs.append(chunk)

            m = meta_to_dict(item.meta)
            # añade campos útiles y de seguridad
            m.update({
                "doc_id": item.doc_id,
                "chunk_index": i,
            })
            metadatas.append(m)

    if not ids:
        raise HTTPException(status_code=400, detail="No hay texto para ingerir.")

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    return {"ingested_chunks": len(ids), "docs": len(items)}

@app.get("/docs")
def list_docs() -> List[str]:
    """Lista nombres de ficheros guardados en ./data (no Chroma)."""
    return sorted([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])

@app.get("/docs/{doc_id}")
def get_doc(doc_id: str):
    """Devuelve el texto de un .txt en ./data; si es PDF, extrae su texto."""
    path = os.path.join(DATA_DIR, doc_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No existe el documento")

    if doc_id.lower().endswith(".pdf"):
        text = ""
        with open(path, "rb") as f:
            reader = PdfReader(BytesIO(f.read()))
            for page in reader.pages:
                text += page.extract_text() or ""
        return {"doc_id": doc_id, "text": clean_text(text)}
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return {"doc_id": doc_id, "text": f.read()}

@app.delete("/docs/{doc_id}")
def delete_doc(doc_id: str):
    """Borra un fichero de ./data (no borra los embeddings del vector DB)."""
    path = os.path.join(DATA_DIR, doc_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No existe el documento")
    os.remove(path)
    return {"deleted": doc_id}

@app.get("/categories")
def list_categories() -> List[str]:
    """Categorías detectadas en metadatos."""
    # Consulta rápida de hasta N elementos para agrupar categorías
    res = collection.get(limit=10000)
    cats = set()
    for md in (res.get("metadatas") or []):
        if md and isinstance(md, dict):
            c = md.get("category")
            if isinstance(c, str):
                cats.add(c)
    return sorted(cats)

@app.get("/cases")
def list_cases() -> List[str]:
    """IDs de caso detectados en metadatos."""
    res = collection.get(limit=10000)
    cases = set()
    for md in (res.get("metadatas") or []):
        if md and isinstance(md, dict):
            cid = md.get("case_id")
            if isinstance(cid, str):
                cases.add(cid)
    return sorted(cases)

@app.post("/query", response_model=QueryResponse)
def query(q: QueryRequest):
    """Consulta semántica contra la base vectorial, con filtros opcionales (category, case_id…)."""
    where = q.filters if q.filters else None
    res = collection.query(
        query_texts=[q.query],
        n_results=max(1, min(q.top_k, 20)),
        where=where
    )

    results: List[QueryChunk] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    scores = res.get("distances", [[]])[0] or res.get("embeddings", [[]])  # compat fallback

    for text, meta, score in zip(docs, metas, scores):
        if not isinstance(meta, dict):
            meta = {}
        results.append(QueryChunk(
            doc_id=str(meta.get("doc_id", "")),
            text=text or "",
            score=float(score) if isinstance(score, (int, float)) else 0.0,
            meta=meta
        ))
    return QueryResponse(results=results)
