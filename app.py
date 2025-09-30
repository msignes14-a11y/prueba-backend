from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from chromadb import PersistentClient

# === CONFIGURACIÓN DE CHROMA ===
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="sibila",
    metadata={"hnsw:space": "cosine"}
)

# === CONFIGURACIÓN DE FASTAPI ===
app = FastAPI(title="SIBILA - Backend Jurídico IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes restringir a tu dominio WordPress más adelante
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODELOS ===
class MetaData(BaseModel):
    filename: str
    category: Optional[str] = None
    case_id: Optional[str] = None

class IngestDoc(BaseModel):
    doc_id: str
    text: str
    meta: MetaData

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    filters: Optional[dict] = None

# === ENDPOINT DE TEST ===
@app.get("/")
def read_root():
    return {"message": "🚀 SIBILA API desplegada correctamente"}

# === INGESTA DE DOCUMENTOS (texto plano) ===
@app.post("/ingest")
def ingest_docs(docs: List[IngestDoc]):
    try:
        for item in docs:
            collection.add(
                documents=[item.text],
                metadatas=[item.meta.dict()],
                ids=[item.doc_id]
            )
        return {"status": "ok", "ingested_docs": len(docs)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# === BÚSQUEDA SEMÁNTICA ===
@app.post("/query")
def query_docs(req: QueryRequest):
    try:
        results = collection.query(
            query_texts=[req.query],
            n_results=req.top_k,
            where=req.filters or {}
        )
        output = []
        for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            output.append({
                "text": doc,
                "meta": meta,
                "score": score
            })
        return {"results": output}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# === SUBIDA DE ARCHIVOS (PDF/TXT) ===
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), category: Optional[str] = Form(None), case_id: Optional[str] = Form(None)):
    try:
        save_path = os.path.join("data", file.filename)
        os.makedirs("data", exist_ok=True)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Si es .txt, lo ingerimos directamente
        if file.filename.endswith(".txt"):
            with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            collection.add(
                documents=[text],
                metadatas=[{"filename": file.filename, "category": category, "case_id": case_id}],
                ids=[file.filename]
            )

        # Si es .pdf (en el futuro añadimos parser PDF si quieres)
        return {"status": "ok", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
