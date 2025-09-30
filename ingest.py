import os, json, re
from typing import List, Dict
from dotenv import load_dotenv
import requests

# Intentar lector PDF desde PyPDF2 o, si no, pypdf (si faltan, los TXT seguirán funcionando)
try:
    from PyPDF2 import PdfReader
except Exception:
    try:
        from pypdf import PdfReader
    except Exception:
        PdfReader = None

from models import JurisMeta

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()
API_BASE   = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
API_URL    = f"{API_BASE}/ingest"
TOKEN      = os.getenv("API_BEARER_TOKEN")
DATA_DIR   = os.getenv("INGEST_FOLDER", "./data")

# ── Utilidad de limpieza ───────────────────────────────────────────────────────
def clean_text(t: str) -> str:
    t = t.replace("\x00", " ").replace("\u0000", " ")
    t = re.sub(r"[\r\t]", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ── Lectores ───────────────────────────────────────────────────────────────────
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return clean_text(f.read())

def read_pdf(path: str) -> str:
    if PdfReader is None:
        print(f"⚠️  No hay librería PDF instalada; se omite: {os.path.basename(path)}")
        return ""
    try:
        reader = PdfReader(path)
        pages = [(p.extract_text() or "") for p in reader.pages]
        return clean_text("\n".join(pages))
    except Exception as e:
        print(f"⚠️  Error leyendo PDF {os.path.basename(path)}: {e}")
        return ""

# ── Construcción de items ──────────────────────────────────────────────────────
def iter_files(folder: str):
    for root, _, files in os.walk(folder):
        for name in files:
            yield os.path.join(root, name)

def build_items(folder: str):
    items = []
    for fp in iter_files(folder):
        base, ext = os.path.splitext(fp)
        name      = os.path.basename(fp)
        ext       = ext.lower()

        if ext == ".txt":
            text = read_txt(fp)
        elif ext == ".pdf":
            text = read_pdf(fp)
        else:
            continue

        if not text.strip():
            print(f"⚠️  {name} sin texto extraíble; se omite.")
            continue

        # Metadatos (si existe <archivo>.meta.json lo usamos)
        meta = JurisMeta(filename=name)
        meta_path = base + ".meta.json"
        if os.path.exists(meta_path):
            try:
                data = json.load(open(meta_path, "r", encoding="utf-8"))
                # filename siempre presente
                data.setdefault("filename", name)
                meta = JurisMeta(**data)
            except Exception as e:
                print(f"⚠️  {os.path.basename(meta_path)} inválido: {e}")

        # Quitar claves con None (Chromadb no admite None)
        meta_clean = {k: v for k, v in meta.model_dump().items() if v is not None}

        items.append({
            "doc_id": os.path.basename(base),
            "text":   text,
            "meta":   meta_clean
        })
    return items

# ── Envío al backend (prueba 2 formatos) ───────────────────────────────────────
def send_items(items: list):
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    # formato A: lista directa
    r = requests.post(API_URL, headers=headers, json=items, timeout=120)
    print("→ POST /ingest (lista) →", r.status_code, r.text[:300])
    if r.status_code >= 400:
        # formato B: con envoltorio
        r = requests.post(API_URL, headers=headers, json={"items": items}, timeout=120)
        print("→ POST /ingest ({'items':[...]}) →", r.status_code, r.text[:300])

    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Carpeta de ingesta:", os.path.abspath(DATA_DIR))
    if not os.path.isdir(DATA_DIR):
        print("⚠️  No existe la carpeta de datos:", DATA_DIR)
    else:
        docs = build_items(DATA_DIR)
        if not docs:
            print("⚠️  No hay documentos válidos en", DATA_DIR)
        else:
            res = send_items(docs)
            print("Resultado:", res)
