import json
import os
from typing import Optional

from fastapi import FastAPI, Query

from .engine import CodeSearchEngine

app = FastAPI(title="Embeddings Code Search")

# Lazy-initialized global engine (avoid model load at import time)
_engine: Optional[CodeSearchEngine] = None

INDEX_PATH = os.environ.get("INDEX_PATH", "data/demo_usearch.idx")
DOCS_PATH = os.environ.get("DOCS_PATH", "data/demo_docs.jsonl")
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def _load_engine():
    global _engine
    if _engine is None:
        _engine = CodeSearchEngine(model_name=MODEL_NAME)
        # Load docs
        docs = [json.loads(line) for line in open(DOCS_PATH)]
        _engine.ingest(docs)
    # Load index (optional: replace in-memory index with persisted one if exists)
    if os.path.exists(INDEX_PATH):
        _engine.idx.load(INDEX_PATH)
    return _engine


@app.get("/search")
def search(query: str = Query(...), k: int = Query(10)):
    eng = _load_engine()
    return {"query": query, "results": eng.search(query, k=k)}
