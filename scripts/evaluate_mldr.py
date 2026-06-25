from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from graphbert.mldr import load_mldr, merge_topk, ndcg_at_k, recall_at_k, relevance_from_examples
from graphbert.retrieval import LongContextRetriever, encode_single_texts, load_retrieval_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single-vector retriever on English MLDR.")
    parser.add_argument("--checkpoint", required=True, help="Directory containing retrieval_model.pt.")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-dir", default="outputs/mldr-single-vector")
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--corpus-batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--document-max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-corpus-documents", type=int, default=None, help="Debug-only corpus limit.")
    return parser.parse_args()


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = LongContextRetriever.load(args.checkpoint).to(device).eval()
    tokenizer = load_retrieval_tokenizer(args.checkpoint)
    query_max_length = args.query_max_length
    document_max_length = args.document_max_length

    query_rows = list(load_mldr(args.language, args.split))
    query_ids = [row["query_id"] for row in query_rows]
    query_texts = [row["query"] for row in query_rows]
    relevance = relevance_from_examples(query_rows)

    query_parts = []
    for texts in tqdm(list(batched(query_texts, args.query_batch_size)), desc="Encoding queries"):
        query_parts.append(encode_single_texts(model, tokenizer, texts, query_max_length, device))
    query_embeddings = torch.cat(query_parts, dim=0)

    top_scores = torch.empty((len(query_ids), 0), device=device)
    top_indices = torch.empty((len(query_ids), 0), dtype=torch.long, device=device)
    document_ids = []
    corpus = load_mldr(args.language, "corpus", streaming=True)
    corpus_batch = []

    for row in tqdm(corpus, desc="Encoding MLDR corpus", total=args.max_corpus_documents or 200000):
        if args.max_corpus_documents is not None and len(document_ids) + len(corpus_batch) >= args.max_corpus_documents:
            break
        corpus_batch.append(row)
        if len(corpus_batch) < args.corpus_batch_size:
            continue
        texts = [item["text"] for item in corpus_batch]
        embeddings = encode_single_texts(model, tokenizer, texts, document_max_length, device)
        scores = query_embeddings @ embeddings.transpose(0, 1)
        offset = len(document_ids)
        new_indices = torch.arange(offset, offset + len(corpus_batch), device=device).expand(len(query_ids), -1)
        top_scores, top_indices = merge_topk(top_scores, top_indices, scores, new_indices, args.top_k)
        document_ids.extend(item["docid"] for item in corpus_batch)
        corpus_batch = []

    if corpus_batch:
        texts = [item["text"] for item in corpus_batch]
        embeddings = encode_single_texts(model, tokenizer, texts, document_max_length, device)
        scores = query_embeddings @ embeddings.transpose(0, 1)
        offset = len(document_ids)
        new_indices = torch.arange(offset, offset + len(corpus_batch), device=device).expand(len(query_ids), -1)
        top_scores, top_indices = merge_topk(top_scores, top_indices, scores, new_indices, args.top_k)
        document_ids.extend(item["docid"] for item in corpus_batch)

    rankings = {
        query_id: [document_ids[index] for index in row]
        for query_id, row in zip(query_ids, top_indices.cpu().tolist())
    }
    metrics = {
        "ndcg_at_10": ndcg_at_k(rankings, relevance, 10),
        "recall_at_100": recall_at_k(rankings, relevance, 100),
        "queries": len(query_ids),
        "corpus_documents": len(document_ids),
        "query_max_length": query_max_length,
        "document_max_length": document_max_length,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "run.trec").open("w", encoding="utf-8") as handle:
        score_rows = top_scores.cpu().tolist()
        for query_id, docs, scores in zip(query_ids, rankings.values(), score_rows):
            for rank, (doc_id, score) in enumerate(zip(docs, scores), start=1):
                handle.write(f"{query_id} Q0 {doc_id} {rank} {score:.8f} longformer-appnp\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
