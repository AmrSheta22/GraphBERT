from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from torch.utils.data import Dataset


FILLER_WORDS = (
    "archive ordinary background report history science culture music travel public record "
    "collection library account summary detail note chapter index reference material"
).split()


def synthetic_code(index: int, seed: int) -> str:
    """Return a deterministic, non-semantic key that is stable across runs."""
    rng = random.Random(seed * 1_000_003 + index)
    return f"key-{rng.randrange(10_000_000, 99_999_999)}-{rng.randrange(1000, 9999)}"


def build_haystack(
    code: str,
    document_words: int,
    position_fraction: float,
    sample_id: int = 0,
) -> str:
    """Create a controlled long document with one passkey-like needle."""
    if document_words < 16:
        raise ValueError("document_words must be at least 16")
    if not 0.0 <= position_fraction <= 1.0:
        raise ValueError("position_fraction must be in [0, 1]")

    offset = sample_id % len(FILLER_WORDS)
    filler = [FILLER_WORDS[(offset + i) % len(FILLER_WORDS)] for i in range(document_words)]
    insertion = round(position_fraction * len(filler))
    needle = ["personalized", "passkey", "is", code]
    words = filler[:insertion] + needle + filler[insertion:]
    return " ".join(words) + "."


def build_query(code: str) -> str:
    return f"Find the archive whose personalized passkey is {code}."


def make_synthetic_triplet(
    index: int,
    document_words: int,
    seed: int,
    position_fractions: Sequence[float] = (0.1, 0.5, 0.9),
) -> Dict[str, str | float]:
    if not position_fractions:
        raise ValueError("position_fractions must not be empty")
    position = float(position_fractions[index % len(position_fractions)])
    positive_code = synthetic_code(index, seed)
    negative_code = synthetic_code(index + 10_000_000, seed)
    return {
        "query": build_query(positive_code),
        "positive": build_haystack(positive_code, document_words, position, index),
        "negative": build_haystack(negative_code, document_words, position, index + 1),
        "position_fraction": position,
    }


class SyntheticNeedleDataset(Dataset):
    """Small deterministic contrastive dataset inspired by LongEmbed passkey retrieval."""

    def __init__(
        self,
        num_samples: int,
        document_words: int,
        seed: int,
        position_fractions: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = num_samples
        self.document_words = document_words
        self.seed = seed
        self.position_fractions = tuple(position_fractions)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, str | float]:
        return make_synthetic_triplet(
            index,
            self.document_words,
            self.seed,
            self.position_fractions,
        )


@dataclass
class SyntheticBenchmark:
    query_ids: List[str]
    queries: List[str]
    document_ids: List[str]
    documents: List[str]
    relevance: Dict[str, Set[str]]
    query_positions: Dict[str, float]


def build_synthetic_benchmark(
    num_queries: int,
    document_words: int,
    seed: int,
    position_fractions: Sequence[float] = (0.1, 0.5, 0.9),
) -> SyntheticBenchmark:
    """Build one relevant and one distractor document per query."""
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")
    if not position_fractions:
        raise ValueError("position_fractions must not be empty")

    query_ids: List[str] = []
    queries: List[str] = []
    document_ids: List[str] = []
    documents: List[str] = []
    relevance: Dict[str, Set[str]] = {}
    query_positions: Dict[str, float] = {}

    for index in range(num_queries):
        query_id = f"q{index:05d}"
        positive_id = f"d{index:05d}-positive"
        distractor_id = f"d{index:05d}-distractor"
        position = float(position_fractions[index % len(position_fractions)])
        positive_code = synthetic_code(index, seed)
        distractor_code = synthetic_code(index + 10_000_000, seed)

        query_ids.append(query_id)
        queries.append(build_query(positive_code))
        document_ids.extend((positive_id, distractor_id))
        documents.extend(
            (
                build_haystack(positive_code, document_words, position, index),
                build_haystack(distractor_code, document_words, position, index + 1),
            )
        )
        relevance[query_id] = {positive_id}
        query_positions[query_id] = position

    return SyntheticBenchmark(
        query_ids=query_ids,
        queries=queries,
        document_ids=document_ids,
        documents=documents,
        relevance=relevance,
        query_positions=query_positions,
    )
