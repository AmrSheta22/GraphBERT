from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import torch
from datasets import load_dataset

MLDR_BASE_URL = "https://huggingface.co/datasets/Shitao/MLDR/resolve/main"


def mldr_url(language: str, split: str) -> str:
    if split == "corpus":
        return f"{MLDR_BASE_URL}/mldr-v1.0-{language}/corpus.jsonl.gz"
    return f"{MLDR_BASE_URL}/mldr-v1.0-{language}/{split}.jsonl.gz"


def load_mldr(language: str = "en", split: str = "test", streaming: bool = False):
    return load_dataset("json", data_files=mldr_url(language, split), split="train", streaming=streaming)


def relevance_from_examples(examples: Iterable[Mapping]) -> Dict[str, Set[str]]:
    return {
        example["query_id"]: {passage["docid"] for passage in example["positive_passages"]}
        for example in examples
    }


def ndcg_at_k(
    rankings: Mapping[str, Sequence[str]],
    relevance: Mapping[str, Set[str]],
    k: int = 10,
) -> float:
    values = []
    for query_id, relevant_docs in relevance.items():
        ranked = rankings.get(query_id, ())[:k]
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, doc_id in enumerate(ranked)
            if doc_id in relevant_docs
        )
        ideal_hits = min(len(relevant_docs), k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        values.append(dcg / idcg if idcg else 0.0)
    return sum(values) / max(len(values), 1)


def recall_at_k(
    rankings: Mapping[str, Sequence[str]],
    relevance: Mapping[str, Set[str]],
    k: int = 100,
) -> float:
    values = []
    for query_id, relevant_docs in relevance.items():
        retrieved = set(rankings.get(query_id, ())[:k])
        values.append(len(retrieved & relevant_docs) / max(len(relevant_docs), 1))
    return sum(values) / max(len(values), 1)


def merge_topk(
    current_scores: torch.Tensor,
    current_indices: torch.Tensor,
    new_scores: torch.Tensor,
    new_indices: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.cat([current_scores, new_scores], dim=1)
    indices = torch.cat([current_indices, new_indices], dim=1)
    top_scores, positions = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
    return top_scores, torch.gather(indices, 1, positions)