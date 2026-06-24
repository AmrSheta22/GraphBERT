from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from graphbert.mldr import load_mldr, ndcg_at_k, recall_at_k, relevance_from_examples
from graphbert.retrieval import LongContextRetriever, encode_token_texts, load_retrieval_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an out-of-domain ColBERT checkpoint on English MLDR.")
    parser.add_argument("--checkpoint", required=True, help="MS MARCO-trained ColBERT retrieval checkpoint.")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-dir", default="outputs/mldr-colbert")
    parser.add_argument("--index-folder", default="outputs/mldr-colbert/index")
    parser.add_argument("--index-name", default="mldr-en")
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--corpus-batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--index-add-batch-size", type=int, default=128)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--document-max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-corpus-documents", type=int, default=None, help="Debug-only corpus limit.")
    return parser.parse_args()


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def result_doc_id(item):
    if isinstance(item, dict):
        return str(item.get("id", item.get("document_id", item.get("doc_id"))))
    if isinstance(item, (tuple, list)):
        return str(item[0])
    return str(item)


def main():
    args = parse_args()
    try:
        from pylate import indexes, retrieve
    except ImportError as error:
        raise SystemExit("Install the optional ColBERT dependency with: pip install pylate") from error

    device = torch.device(args.device)
    model = LongContextRetriever.load(args.checkpoint).to(device).eval()
    if model.retrieval_config.projection_dim <= 0:
        raise ValueError("ColBERT evaluation requires a checkpoint trained with --architecture colbert.")
    tokenizer = load_retrieval_tokenizer(args.checkpoint)
    query_max_length = args.query_max_length
    document_max_length = args.document_max_length

    index = indexes.PLAID(
        index_folder=args.index_folder,
        index_name=args.index_name,
        override=True,
    )
    corpus = load_mldr(args.language, "corpus", streaming=True)
    encode_rows = []
    pending_ids = []
    pending_embeddings = []
    indexed = 0

    def flush_index():
        nonlocal indexed, pending_ids, pending_embeddings
        if not pending_ids:
            return
        index.add_documents(
            documents_ids=pending_ids,
            documents_embeddings=pending_embeddings,
        )
        indexed += len(pending_ids)
        pending_ids = []
        pending_embeddings = []

    for row in tqdm(corpus, desc="Indexing MLDR corpus", total=args.max_corpus_documents or 200000):
        seen = indexed + len(pending_ids) + len(encode_rows)
        if args.max_corpus_documents is not None and seen >= args.max_corpus_documents:
            break
        encode_rows.append(row)
        if len(encode_rows) < args.corpus_batch_size:
            continue
        embeddings = encode_token_texts(
            model, tokenizer, [item["text"] for item in encode_rows], document_max_length, device
        )
        pending_ids.extend(item["docid"] for item in encode_rows)
        pending_embeddings.extend(embeddings)
        encode_rows = []
        if len(pending_ids) >= args.index_add_batch_size:
            flush_index()
    if encode_rows:
        embeddings = encode_token_texts(
            model, tokenizer, [item["text"] for item in encode_rows], document_max_length, device
        )
        pending_ids.extend(item["docid"] for item in encode_rows)
        pending_embeddings.extend(embeddings)
    flush_index()

    query_rows = list(load_mldr(args.language, args.split))
    query_ids = [row["query_id"] for row in query_rows]
    relevance = relevance_from_examples(query_rows)
    retriever = retrieve.ColBERT(index=index)
    rankings = {}
    for rows in tqdm(list(batched(query_rows, args.query_batch_size)), desc="Retrieving"):
        query_embeddings = encode_token_texts(
            model, tokenizer, [row["query"] for row in rows], query_max_length, device
        )
        results = retriever.retrieve(
            queries_embeddings=[embedding.numpy() for embedding in query_embeddings],
            k=args.top_k,
        )
        for row, result in zip(rows, results):
            rankings[row["query_id"]] = [result_doc_id(item) for item in result]

    metrics = {
        "ndcg_at_10": ndcg_at_k(rankings, relevance, 10),
        "recall_at_100": recall_at_k(rankings, relevance, 100),
        "queries": len(query_ids),
        "corpus_documents": indexed,
        "query_max_length": query_max_length,
        "document_max_length": document_max_length,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "rankings.json").open("w", encoding="utf-8") as handle:
        json.dump(rankings, handle)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()