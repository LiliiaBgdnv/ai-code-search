import argparse
import json
import math

import numpy as np

from src.engine import CodeSearchEngine

"""
Evaluate retrieval with Recall@k, MRR@k, NDCG@k.

Inputs are JSONL:
- corpus  : {doc_id?, text|code, func_name?, lang?}
- queries : {qid, query}
- qrels   : {qid, doc_id, label}  (label > 0 means relevant)
"""


def read_jsonl(path: str) -> list[dict]:
    """Read JSONL into list of dicts. Skips empty lines."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def dcg(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain."""
    rel = relevances[:k]
    return sum(((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel)))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """Normalized DCG at k."""
    idcg = dcg(sorted(relevances, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def recall_at_k(relevances: list[int], k: int, total_rel: int) -> float | None:
    """Recall@k. If no relevant docs exist for query, return None."""
    if total_rel == 0:
        return None
    return sum(1 for r in relevances[:k] if r > 0) / float(total_rel)


def mrr_at_k(relevances: list[int], k: int) -> float:
    """Reciprocal rank of the first relevant item within top-k."""
    for i, r in enumerate(relevances[:k], start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def evaluate(
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
    index_path: str | None,  # note: not used because, embed in-memory here
    model_name: str,
    k: int = 10,
) -> dict[str, float]:
    """Run retrieval for all queries and aggregate metrics."""

    corpus = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    qrels = read_jsonl(qrels_path)

    qrel_map: dict[str, dict[str, int]] = {}
    for r in qrels:
        if int(r.get("label", 0)) > 0:
            qid = str(r["qid"])
            did = str(r["doc_id"])
            qrel_map.setdefault(qid, {})[did] = 1

    docs = []
    for i, d in enumerate(corpus):
        did = str(d.get("doc_id", i))
        txt = d.get("text") or d.get("code") or ""
        docs.append(
            {
                "id": did,
                "text": txt,  # engine expects 'text' or 'code'; set 'text' here
                "meta": {
                    "func_name": d.get("func_name"),
                    "lang": d.get("lang", "python"),
                },
            }
        )

    # In-memory engine (fresh embeddings for corpus)
    engine = CodeSearchEngine(model_name=model_name)
    engine.ingest(docs)

    recall_vals: list[float] = []
    mrr_vals: list[float] = []
    ndcg_vals: list[float] = []

    for q in queries:
        qid = str(q["qid"])
        gold = qrel_map.get(qid, {})
        total_rel = len(gold)
        if total_rel == 0:
            continue

        res = engine.search(q["query"], k=k)
        relevances = [1 if gold.get(r["doc_id"]) else 0 for r in res]

        r = recall_at_k(relevances, k, total_rel)
        if r is not None:
            recall_vals.append(r)
        mrr_vals.append(mrr_at_k(relevances, k))
        ndcg_vals.append(ndcg_at_k(relevances, k))

    def mean_safe(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "Recall@10": mean_safe(recall_vals),
        "MRR@10": mean_safe(mrr_vals),
        "NDCG@10": mean_safe(ndcg_vals),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval metrics")
    p.add_argument("--corpus", required=True, help="JSONL with code/text corpus")
    p.add_argument("--queries", required=True, help="JSONL with queries {qid, query}")
    p.add_argument(
        "--qrels", required=True, help="JSONL with qrels {qid, doc_id, label>0}"
    )
    p.add_argument(
        "--index",
        required=False,
        default=None,
        help="(unused here) Path to saved index if you want to reuse (future use)",
    )
    p.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    p.add_argument("--k", type=int, default=10, help="Top-k to evaluate")
    args = p.parse_args()

    scores = evaluate(
        args.corpus, args.queries, args.qrels, args.index, args.model, args.k
    )
    print(json.dumps(scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
