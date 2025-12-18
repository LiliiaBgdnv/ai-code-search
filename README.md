# Embeddings-based Code Search Backend

This repository contains a **minimal, end-to-end retrieval backend** for embeddings-based code search,
designed as a reusable component for RAG or LLM-powered systems.

- **Part 1**: Embeddings-based search engine (FastAPI + USearch)
- **Part 2**: Evaluation on **CoSQA** (Recall@10, MRR@10, NDCG@10)
- **Part 3**: Optional fine-tuning with **sentence-transformers** (MultipleNegativesRankingLoss)

> CoSQA dataset: 20,604 natural-language queries paired with code snippets  
> (see the ACL 2021 paper and the Hugging Face dataset card).

## Quickstart

## Part 0 — Setup
Clone the repository:
```bash
git clone git@github.com:LiliiaBgdnv/ai-code-search.git
```

Preparation Python 3.11+ recommended
```bash
python -m venv .venv && source .venv/bin/activate
```

Install deps
```bash
cd ai-code-search
pip install -r requirements.txt
```

# Part 1 — Build and run demo search engine
Make demo dataset
```bash
mkdir -p data
cat > data/demo_docs.jsonl <<'EOF'
{"id": 1, "text": "def add(a, b): return a + b", "meta": {"lang": "python"}}
{"id": 2, "text": "def multiply(a, b): return a * b", "meta": {"lang": "python"}}
{"id": 3, "text": "import os; os.mkdir('new_dir')", "meta": {"lang": "python"}}
{"id": 4, "text": "def divide(a, b): return a / b", "meta": {"lang": "python"}}
{"id": 5, "text": "def subtract(a, b): return a - b", "meta": {"lang": "python"}}
EOF
```
```bash
python -m src.build_index \
  --docs data/demo_docs.jsonl \
  --index data/demo_usearch.idx \
  --model sentence-transformers/all-MiniLM-L6-v2
```

## Start FastAPI server
```bash
uvicorn src.search_api:app --reload --port 8000
```
then open: [http://127.0.0.1:8000/docs]  and try /search?query="create directory"

# Part 2 — Prepare and evaluate on CoSQA

## Prepare dataset
```bash
python -m src.prepare_cosqa --out_dir data/cosqa
```

This creates:
```bash
data/cosqa/
  corpus.jsonl
  queries.jsonl
  qrels.jsonl
  train_pairs.jsonl
  val_pairs.jsonl
```

## Evaluate baseline model
```bash
python -m src.eval \
  --corpus data/cosqa/corpus.jsonl \
  --queries data/cosqa/queries.jsonl \
  --qrels data/cosqa/qrels.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2
```
You’ll get metrics:
*Recall@10*, *MRR@10*, *NDCG@10* — all averaged across queries.

# Part 3 — Fine-tuning
Quick 1-epoch fine-tune on positive query–code pairs:
```bash
python -m src.train \
  --train_pairs data/cosqa/train_pairs.jsonl \
  --val_pairs data/cosqa/val_pairs.jsonl \
  --epochs 1 \
  --batch_size 64 \
  --lr 2e-5 \
  --out_dir models/cosqa-mini
```
Then rebuild your index with the fine-tuned model and re-run evaluation to compare metrics.