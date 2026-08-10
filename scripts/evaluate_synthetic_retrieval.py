from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from graphbert.mldr import ndcg_at_k, recall_at_k
from graphbert.retrieval import LongContextRetriever, encode_single_texts, load_retrieval_tokenizer
from graphbert.synthetic_retrieval import build_synthetic_benchmark, synthetic_code


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate distance-controlled synthetic long-context retrieval without external datasets."
    )
    parser.add_argument("--checkpoint", required=True, help="Directory containing retrieval_model.pt.")
    parser.add_argument("--output-dir", default="outputs/synthetic-long-retrieval-eval")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--num-queries", type=int, default=60)
    parser.add_argument("--positions", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument(
        "--filler-ratio",
        type=float,
        default=0.55,
        help="Filler words per tokenizer length; 0.55 keeps the late needle before truncation for Longformer BPE.",
    )
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--corpus-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def contains_subsequence(values, target):
    width = len(target)
    return any(values[start : start + width] == target for start in range(len(values) - width + 1))


def needle_retention_rate(tokenizer, benchmark, max_length, seed):
    retained = 0
    for index in range(len(benchmark.query_ids)):
        document = benchmark.documents[index * 2]
        code = synthetic_code(index, seed)
        document_ids = tokenizer(document, truncation=True, max_length=max_length)["input_ids"]
        code_ids = tokenizer(code, add_special_tokens=False)["input_ids"]
        retained += contains_subsequence(document_ids, code_ids)
    return retained / len(benchmark.query_ids)


def evaluate_length(model, tokenizer, args, max_length):
    document_words = max(16, int(max_length * args.filler_ratio))
    benchmark = build_synthetic_benchmark(
        num_queries=args.num_queries,
        document_words=document_words,
        seed=args.seed,
        position_fractions=args.positions,
    )
    device = torch.device(args.device)

    query_parts = []
    for texts in batched(benchmark.queries, args.query_batch_size):
        query_parts.append(encode_single_texts(model, tokenizer, texts, args.query_max_length, device))
    query_embeddings = torch.cat(query_parts, dim=0)

    document_parts = []
    batches = list(batched(benchmark.documents, args.corpus_batch_size))
    for texts in tqdm(batches, desc=f"Encoding {max_length}-token corpus"):
        document_parts.append(encode_single_texts(model, tokenizer, texts, max_length, device))
    document_embeddings = torch.cat(document_parts, dim=0)
    scores = query_embeddings @ document_embeddings.transpose(0, 1)
    ranked_indices = scores.argsort(dim=1, descending=True).cpu().tolist()
    rankings = {
        query_id: [benchmark.document_ids[index] for index in row]
        for query_id, row in zip(benchmark.query_ids, ranked_indices)
    }

    by_position = {}
    for position in args.positions:
        ids = [
            query_id
            for query_id in benchmark.query_ids
            if benchmark.query_positions[query_id] == position
        ]
        hits = sum(rankings[query_id][0] in benchmark.relevance[query_id] for query_id in ids)
        by_position[str(position)] = hits / max(len(ids), 1)

    return {
        "document_max_length": max_length,
        "document_words": document_words,
        "queries": len(benchmark.query_ids),
        "corpus_documents": len(benchmark.document_ids),
        "ndcg_at_10": ndcg_at_k(rankings, benchmark.relevance, 10),
        "recall_at_1": recall_at_k(rankings, benchmark.relevance, 1),
        "top1_accuracy_by_position": by_position,
        "needle_retention_rate": needle_retention_rate(tokenizer, benchmark, max_length, args.seed),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = LongContextRetriever.load(args.checkpoint).to(device).eval()
    tokenizer = load_retrieval_tokenizer(args.checkpoint)

    results = [evaluate_length(model, tokenizer, args, length) for length in args.lengths]
    metrics = {
        "benchmark": "LongEmbed/RULER-inspired personalized passkey retrieval",
        "checkpoint": args.checkpoint,
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
