from __future__ import annotations

from itertools import chain
from typing import Dict

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from graphbert.config import DatasetConfig


def load_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def load_mlm_dataset(dataset_config: DatasetConfig) -> DatasetDict:
    raw = load_dataset(dataset_config.name, dataset_config.config_name)
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=dataset_config.validation_split_percentage / 100.0)
        raw = DatasetDict(train=split["train"], validation=split["test"])
    return raw


def tokenize_and_group(raw_datasets: DatasetDict, tokenizer, dataset_config: DatasetConfig) -> DatasetDict:
    text_column = dataset_config.text_column
    max_seq_length = min(dataset_config.max_seq_length, tokenizer.model_max_length)

    if dataset_config.line_by_line:
        def tokenize_line_by_line(examples):
            lines = [line for line in examples[text_column] if line and not line.isspace()]
            return tokenizer(lines, padding=False, truncation=True, max_length=max_seq_length)

        return raw_datasets.map(
            tokenize_line_by_line,
            batched=True,
            num_proc=dataset_config.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing line-by-line",
        )

    def tokenize_function(examples):
        return tokenizer(examples[text_column], return_special_tokens_mask=True)

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=dataset_config.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing dataset",
    )

    def group_texts(examples: Dict[str, list]):
        concatenated = {key: list(chain(*examples[key])) for key in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            key: [values[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for key, values in concatenated.items()
        }
        return result

    return tokenized.map(
        group_texts,
        batched=True,
        num_proc=dataset_config.preprocessing_num_workers,
        desc=f"Grouping texts into blocks of {max_seq_length}",
    )


def build_mlm_collator(tokenizer, mlm_probability: float):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
