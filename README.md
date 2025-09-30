# JurisRAG Backend (FastAPI + Chroma)

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Ingesta
Coloca PDFs/TXTs en `./data` (opcional: `.meta.json` con metadatos). Luego:
```bash
export INGEST_FOLDER=./data
python ingest.py
```

## Query
`POST /query` body:
```json
{"query":"cláusula suelo AP Valencia estimadas","top_k":8,"filters":{"tribunal":"AP Valencia"}}
```
