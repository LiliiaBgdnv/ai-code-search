import argparse
import json
import os
from typing import Optional

import numpy as np

from src.embeddings import Embedder
from src.index_usearch import VectorIndex


class CodeSearchEngine:
    """Simple code/text search engine:
    - embeds input docs
    - builds USearch ANN index
    - supports top-k cosine search
    """

    def __init__(self, model_name: str, index_params: Optional[dict] = None):
        self.embedder = Embedder(model_name=model_name)
        # Create ANN index with correct dimensionality
        self.idx = VectorIndex(ndim=self.embedder.ndim, **(index_params or {}))
        self.docs: list[dict] = []
        # map internal position -> original doc id
        self.id2pos = {}

    def ingest(self, docs: list[dict]):
        """Add documents to index.

        Expected keys per doc:
          - id (or doc_id)
          - text or code (we will index one string)
          - meta (optional)
        """
        # backfill id if missing
        for i, d in enumerate(docs):
            if "id" not in d:
                d["id"] = d.get("doc_id", i)

        # choose content field (text>code). Raise if both missing
        texts: list[str] = []
        for d in docs:
            if "text" in d:
                texts.append(d["text"])
            elif "code" in d:
                texts.append(d["code"])
            else:
                raise KeyError(f"Missing 'text' or 'code' in doc {d.get('id')}")

        # embed and add to ANN
        self.docs = docs
        vectors = self.embedder.encode(texts)
        ids = np.arange(len(docs))
        self.id2pos = {i: d["id"] for i, d in enumerate(docs)}
        self.idx.add(ids=ids, vectors=vectors)

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k matches for query with similarity scores.

        Score = 1 - cosine_distance (so higher is better).
        """
        qv = self.embedder.encode([query])
        labels, dists = self.idx.search(qv, k=k)
        labels = labels[0]
        dists = dists[0]

        out: list[dict] = []
        for rank, (lbl, dist) in enumerate(zip(labels, dists), start=1):
            if lbl == -1:
                # no more neighbors
                continue
            doc = self.docs[int(lbl)]
            out.append(
                {
                    "rank": rank,
                    "doc_id": doc["id"],
                    "score": float(1.0 - dist),  # cos distance -> similarity-like
                    "text": doc.get("text") or doc.get("code", ""),
                    "meta": doc.get("meta", {}),
                }
            )
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple code search CLI")
    parser.add_argument(
        "--model", type=str, required=True, help="Sentence-Transformers model"
    )
    parser.add_argument(
        "--index", type=str, required=True, help="Path to existing USearch index"
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Query string to search"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    eng = CodeSearchEngine(model_name=args.model)

    # For demo CLI we use a fixed small doc file; index must already exist
    DOCS_PATH = "data/demo_docs.jsonl"
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"{DOCS_PATH} not found")

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    eng.ingest(docs)

    # Load prebuilt index (created by scripts/build_index.py)
    eng.idx.load(args.index)

    results = eng.search(args.query, k=args.k)
    print(json.dumps(results, indent=2, ensure_ascii=False))
