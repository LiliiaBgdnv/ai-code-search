import argparse
import json
import os
from typing import List

import torch
from sentence_transformers import InputExample, SentenceTransformer, evaluation, losses
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_pairs(path: str, use_only_positive: bool = True) -> list[InputExample]:
    rows = read_jsonl(path)
    examples = []
    for r in rows:
        q = r.get("query", "")
        c = r.get("code", r.get("text", ""))
        lab = float(r.get("label", 1))
        if use_only_positive and lab <= 0:
            continue
        examples.append(InputExample(texts=[q, c]))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pairs", required=True)
    ap.add_argument("--val_pairs", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--log_path", default="data/train_log.txt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_corpus", type=str, default="data/cosqa/corpus.jsonl")
    ap.add_argument("--val_queries", type=str, default="data/cosqa/queries.jsonl")
    ap.add_argument("--val_qrels", type=str, default="data/cosqa/qrels.jsonl")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    # model & loss
    model = SentenceTransformer(args.model)
    model.to(model.device)

    train_examples = build_pairs(args.train_pairs, use_only_positive=True)
    if len(train_examples) < 2:
        raise ValueError("Not enough training pairs.")

    train_dl = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=min(args.batch_size, max(2, len(train_examples))),
        drop_last=True,
        collate_fn=model.smart_batching_collate,
    )
    print(f"Train examples: {len(train_examples)}, batch_size: {train_dl.batch_size}")

    try:
        corpus = {}
        for r in read_jsonl(args.val_corpus):
            rid = str(r.get("id", r.get("doc_id")))
            corpus[rid] = r.get("code", r.get("text", ""))

        queries = {}
        for r in read_jsonl(args.val_queries):
            qid = str(r.get("qid", r.get("id")))
            queries[qid] = r["query"]

        relevant_docs = {}
        for r in read_jsonl(args.val_qrels):
            qid = str(r.get("qid", r.get("query_id")))
            did = str(r.get("doc_id", r.get("did")))
            if int(r.get("label", 1)) > 0:
                relevant_docs.setdefault(qid, {})[did] = 1

        evaluator = evaluation.InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[10],
            precision_recall_at_k=[10],
            map_at_k=[10],
        )
        print(f"IR evaluator ready: |corpus|={len(corpus)} |queries|={len(queries)}")
    except Exception as e:
        print("IR evaluator init failed:", e)
        evaluator = None

    train_loss = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    steps_per_epoch = max(1, len(train_dl))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    device = model.device
    model.train()

    with open(args.log_path, "w", encoding="utf-8") as lf:
        gstep = 0
        ema = None
        beta = 0.05

        for epoch in range(args.epochs):
            for step, batch in enumerate(train_dl, 1):
                features, labels = batch
                for feat in features:
                    for k, v in feat.items():
                        if hasattr(v, "to"):
                            feat[k] = v.to(device)

                loss_val = train_loss(features, labels)
                loss_val.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                gstep += 1
                loss_float = loss_val.item()
                ema = (
                    loss_float if ema is None else (1 - beta) * ema + beta * loss_float
                )

                if gstep % args.log_every == 0:
                    line = (
                        f"epoch={epoch+1} step={gstep} "
                        f"loss={loss_float:.6f} ema_loss={ema:.6f}\n"
                    )
                    print(line.strip())
                    lf.write(line)
                    lf.flush()

            # simple end-of-epoch evaluation (optional)
            if evaluator is not None:
                print(f"Evaluating after epoch {epoch+1} ...")
                evaluator(model, output_path=None)

    # save model
    model.save(args.out_dir)
    print(f"Saved to: {args.out_dir}")
    print(f"Loss log written to: {args.log_path}")


if __name__ == "__main__":
    main()
