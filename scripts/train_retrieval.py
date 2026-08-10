from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from graphbert.config import load_experiment_config
from graphbert.mldr import load_mldr
from graphbert.retrieval import (
    LongContextRetriever,
    configure_trainable_parameters,
    load_retrieval_tokenizer,
    maxsim_scores,
    tokenize_texts,
)
from graphbert.synthetic_retrieval import SyntheticNeedleDataset

MSMARCO_DATASET = "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Train single- or multi-vector retrieval heads for MLDR evaluation.")
    parser.add_argument("--config", required=True, help="GraphBERT experiment config used to reconstruct the encoder.")
    parser.add_argument("--source-model", required=True, help="MLM checkpoint/HF model, or prior retrieval checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["msmarco", "mldr", "synthetic"], required=True)
    parser.add_argument("--architecture", choices=["single", "colbert"], default="single")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument(
        "--single-projection-dim",
        type=int,
        default=0,
        help="Optional trainable projection for the single-vector architecture; 0 preserves prior behavior.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer steps; 0 uses all data.")
    parser.add_argument("--max-samples", type=int, default=1_250_000)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--document-max-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--distillation-temperature", type=float, default=1.0)
    parser.add_argument("--msmarco-config", default="triplet-hard")
    parser.add_argument("--colbert-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10, help="Print retrieval loss every N optimizer steps.")
    parser.add_argument(
        "--trainable-mode",
        choices=["full", "adapters", "head", "adapters+head"],
        default="full",
        help="Freeze the Longformer backbone for inexpensive adapter/head-only tuning.",
    )
    parser.add_argument("--synthetic-samples", type=int, default=2048)
    parser.add_argument(
        "--synthetic-document-words", type=int, default=700,
        help="Approximate synthetic haystack length before tokenizer truncation.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalize_example(row, stage):
    if stage == "msmarco":
        return {"query": row["query"], "positive": row["positive"], "negative": row["negative"]}
    if stage == "synthetic":
        return {"query": row["query"], "positive": row["positive"], "negative": row["negative"]}
    positives = row["positive_passages"]
    negatives = row["negative_passages"]
    positive = positives[0]["text"]
    negative = random.choice(negatives)["text"] if negatives else positive
    return {"query": row["query"], "positive": positive, "negative": negative}


class ColBERTDistillationDataset(Dataset):
    def __init__(self, max_samples, seed):
        self.training = load_dataset("lightonai/ms-marco-en-bge", "train", split="train")
        self.queries = load_dataset("lightonai/ms-marco-en-bge", "queries", split="train")
        self.documents = load_dataset("lightonai/ms-marco-en-bge", "documents", split="train")
        self.query_text = {row["query_id"]: row["text"] for row in self.queries}
        if max_samples and len(self.training) > max_samples:
            self.training = self.training.shuffle(seed=seed).select(range(max_samples))

    def __len__(self):
        return len(self.training)

    def _document_text(self, doc_id):
        row = self.documents[int(doc_id)]
        if str(row["document_id"]) != str(doc_id):
            raise ValueError("MS MARCO document IDs no longer align with dataset row indices.")
        return row["text"]

    def __getitem__(self, index):
        row = self.training[index]
        return {
            "query": self.query_text[row["query_id"]],
            "documents": [self._document_text(doc_id) for doc_id in row["document_ids"]],
            "teacher_scores": row["scores"],
        }


def load_training_data(args):
    if args.stage == "synthetic":
        return SyntheticNeedleDataset(
            num_samples=min(args.synthetic_samples, args.max_samples) if args.max_samples else args.synthetic_samples,
            document_words=args.synthetic_document_words,
            seed=args.seed,
        )
    if args.stage == "msmarco" and args.architecture == "colbert":
        return ColBERTDistillationDataset(args.max_samples, args.seed)
    if args.stage == "msmarco":
        dataset = load_dataset(MSMARCO_DATASET, args.msmarco_config, split="train")
    else:
        dataset = load_mldr("en", "train")
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))
    return dataset


def collate_rows(rows, stage):
    normalized = [normalize_example(row, stage) for row in rows]
    return {key: [row[key] for row in normalized] for key in ("query", "positive", "negative")}


class RetrievalCollator:
    def __init__(self, stage, architecture, colbert_candidates):
        self.stage = stage
        self.architecture = architecture
        self.colbert_candidates = colbert_candidates

    def __call__(self, rows):
        if self.stage == "msmarco" and self.architecture == "colbert":
            count = self.colbert_candidates
            return {
                "query": [row["query"] for row in rows],
                "documents": [row["documents"][:count] for row in rows],
                "teacher_scores": [row["teacher_scores"][:count] for row in rows],
            }
        return collate_rows(rows, self.stage)

def single_vector_loss(model, tokenizer, batch, args, device):
    queries = model.encode_single(tokenize_texts(tokenizer, batch["query"], args.query_max_length, device))
    positives = model.encode_single(tokenize_texts(tokenizer, batch["positive"], args.document_max_length, device))
    negatives = model.encode_single(tokenize_texts(tokenizer, batch["negative"], args.document_max_length, device))
    candidates = torch.cat([positives, negatives], dim=0)
    logits = queries @ candidates.transpose(0, 1) / args.temperature
    labels = torch.arange(queries.shape[0], device=device)
    return F.cross_entropy(logits, labels)


def colbert_loss(model, tokenizer, batch, args, device):
    query_embeddings, query_mask = model.encode_tokens(
        tokenize_texts(tokenizer, batch["query"], args.query_max_length, device)
    )
    if "documents" in batch:
        group_size = len(batch["documents"][0])
        document_texts = [text for group in batch["documents"] for text in group]
        teacher_scores = torch.tensor(batch["teacher_scores"], dtype=torch.float32, device=device)
        document_embeddings, document_mask = model.encode_tokens(
            tokenize_texts(tokenizer, document_texts, args.document_max_length, device)
        )
        all_scores = maxsim_scores(query_embeddings, query_mask, document_embeddings, document_mask)
        batch_size = query_embeddings.shape[0]
        student_scores = all_scores.view(batch_size, batch_size, group_size)[
            torch.arange(batch_size, device=device), torch.arange(batch_size, device=device)
        ]
        teacher_probabilities = F.softmax(teacher_scores / args.distillation_temperature, dim=-1)
        return F.kl_div(
            F.log_softmax(student_scores / args.distillation_temperature, dim=-1),
            teacher_probabilities,
            reduction="batchmean",
        )

    document_batch = batch["positive"] + batch["negative"]
    document_embeddings, document_mask = model.encode_tokens(
        tokenize_texts(tokenizer, document_batch, args.document_max_length, device)
    )
    logits = maxsim_scores(query_embeddings, query_mask, document_embeddings, document_mask) / args.temperature
    labels = torch.arange(query_embeddings.shape[0], device=device)
    return F.cross_entropy(logits, labels)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    experiment = load_experiment_config(args.config)

    source_path = Path(args.source_model)
    if (source_path / "retrieval_config.json").exists():
        model = LongContextRetriever.load(args.source_model)
        tokenizer = load_retrieval_tokenizer(args.source_model)
        if args.architecture == "colbert" and model.retrieval_config.projection_dim <= 0:
            raise ValueError("The source retrieval checkpoint is not a ColBERT model.")
    else:
        projection_dim = args.projection_dim if args.architecture == "colbert" else args.single_projection_dim
        document_max_length = args.document_max_length or (
            512 if args.stage == "msmarco" else 1024 if args.stage == "synthetic" else 4096
        )
        model = LongContextRetriever.from_source(
            args.source_model,
            experiment.graph,
            pooling=args.pooling,
            projection_dim=projection_dim,
            query_max_length=args.query_max_length,
            document_max_length=document_max_length,
        )
        tokenizer = load_retrieval_tokenizer(args.source_model)
    default_document_length = 180 if args.architecture == "colbert" and args.stage == "msmarco" else (
        512 if args.stage == "msmarco" else 1024 if args.stage == "synthetic" else 4096
    )
    args.document_max_length = args.document_max_length or default_document_length
    if args.architecture == "colbert" and args.stage == "msmarco" and args.query_max_length == 64:
        args.query_max_length = 32
    model.retrieval_config.query_max_length = args.query_max_length
    model.retrieval_config.document_max_length = args.document_max_length
    trainable_summary = configure_trainable_parameters(model, args.trainable_mode)
    print(json.dumps({"trainable": trainable_summary}, indent=2))
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    model.to(device).train()

    dataset = load_training_data(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if args.architecture == "colbert" else args.num_workers,
        collate_fn=RetrievalCollator(args.stage, args.architecture, args.colbert_candidates),
    )
    available_update_steps = math.ceil(len(loader) / args.gradient_accumulation_steps) * args.epochs
    update_steps = min(available_update_steps, args.max_steps) if args.max_steps > 0 else available_update_steps
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(update_steps * args.warmup_ratio),
        num_training_steps=update_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    global_step = 0

    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"{args.stage} epoch {epoch + 1}")
        for step, batch in enumerate(progress, start=1):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                if args.architecture == "single":
                    loss = single_vector_loss(model, tokenizer, batch, args, device)
                else:
                    loss = colbert_loss(model, tokenizer, batch, args, device)
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tqdm.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "optimizer_step": global_step,
                                "loader_step": step,
                                "loss": float(loss.detach().item()),
                                "learning_rate": scheduler.get_last_lr()[0],
                            }
                        )
                    )
            progress.set_postfix(loss=f"{loss.detach().item():.4f}", step=global_step)
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    model.save(args.output_dir, tokenizer)
    with (Path(args.output_dir) / "training_args.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {**vars(args), "trainable_summary": trainable_summary, "completed_steps": global_step},
            handle,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
