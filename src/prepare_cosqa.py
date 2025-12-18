import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def write_jsonl(path, rows):
    """Write list[dict] to JSONL with UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Prepare CoSQA subset in JSONL format")
    p.add_argument(
        "--out_dir", required=True, help="Where to write corpus/queries/qrels"
    )
    p.add_argument(
        "--split", default="validation", help="HF split: train|validation|test"
    )
    p.add_argument("--limit", type=int, default=1000, help="Subset size for quick runs")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load HF dataset
    ds = load_dataset("gonglinyuan/CoSQA", split=args.split)

    # Fields seen on HF: code, code_tokens, doc (web query), docstring_tokens, idx, label
    # We'll build a simple corpus of unique code functions and a query set with qrels.
    corpus, queries, qrels = [], [], []
    doc_id = 0

    for row in tqdm(ds):
        code = row["code"]
        query = row["doc"]  # the web query text
        label = int(row["label"])  # 1 if match, 0 otherwise
        func_name = None

        # crude function name parse (optional)
        if code.strip().startswith("def "):
            try:
                func_name = code.split("def ")[1].split("(")[0].strip()
            except Exception:
                func_name = None

        # Build corpus doc
        corpus.append(
            {"doc_id": doc_id, "code": code, "func_name": func_name, "lang": "python"}
        )

        # Build queries (one per row, pointing to this doc with its label)
        qid = len(queries)
        queries.append({"qid": qid, "query": query})

        # Relevance: same row links to this doc_id with label
        qrels.append({"qid": qid, "doc_id": doc_id, "label": label})
        doc_id += 1

        if args.limit and doc_id >= args.limit:
            break

    # Write outputs
    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "qrels.jsonl", qrels)

    # For fine-tuning, create positive pairs only
    train_pairs = [
        {"query": q["query"], "code": c["code"]}
        for q, c, r in zip(queries, corpus, qrels)
        if r["label"] == 1
    ]

    # 90/10 split
    cut = max(1, int(0.9 * len(train_pairs)))
    write_jsonl(out / "train_pairs.jsonl", train_pairs[:cut])
    write_jsonl(out / "val_pairs.jsonl", train_pairs[cut:])

    print(f"Wrote files into: {out.resolve()}")


if __name__ == "__main__":
    main()
