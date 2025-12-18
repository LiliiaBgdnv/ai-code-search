import argparse
import json

from src.engine import CodeSearchEngine


def read_jsonl(path: str) -> list[dict]:
    """Read JSON Lines file into a list of dicts.

    We skip empty lines to be safe.
    """
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True, help="JSONL with {{id, text, meta}}")
    p.add_argument("--index", required=True, help="Path to save USearch index")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    # Create engine with selected embedding model
    engine = CodeSearchEngine(model_name=args.model)

    # Read documents and ingest to the in-memory index
    docs = read_jsonl(args.docs)
    engine.ingest(docs)

    # Persist the ANN index to disk for fast search later
    engine.idx.save(args.index)
    print(f"Built index with {len(docs)} docs. Saved to {args.index}.")


if __name__ == "__main__":
    main()
