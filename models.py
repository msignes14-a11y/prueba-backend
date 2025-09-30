from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class JurisMeta(BaseModel):
    ecli: Optional[str] = None
    tribunal: Optional[str] = None
    sala: Optional[str] = None
    fecha: Optional[str] = None
    procedimiento: Optional[str] = None
    ponente: Optional[str] = None
    materia: Optional[str] = None
    resultado: Optional[str] = None
    origen: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class IngestItem(BaseModel):
    doc_id: str = Field(...)
    text: str = Field(...)
    meta: JurisMeta

class QueryRequest(BaseModel):
    query: str
    top_k: int = 8
    filters: Optional[Dict[str, Any]] = None

class QueryChunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float
    meta: JurisMeta

class QueryResponse(BaseModel):
    results: List[QueryChunk]
