from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from graphbert.data import build_mlm_collator, load_mlm_dataset, load_tokenizer, tokenize_and_group
from graphbert.metrics import GraphStatsCallback, add_perplexity
from graphbert.modeling import build_graph_bert_for_mlm
from graphbert.utils import load_config_with_overrides, parse_config_args, prepare_reproducibility, save_experiment_config


def build_training_args(config) -> TrainingArguments:
    training = config.training
    kwargs = {
        "output_dir": config.output_dir,
        "overwrite_output_dir": training.overwrite_output_dir,
        "do_train": training.do_train,
        "do_eval": training.do_eval,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "adam_beta1": training.adam_beta1,
        "adam_beta2": training.adam_beta2,
        "adam_epsilon": training.adam_epsilon,
        "max_steps": training.max_steps,
        "warmup_ratio": training.warmup_ratio,
        "logging_steps": training.logging_steps,
        "eval_steps": training.eval_steps,
        "save_steps": training.save_steps,
        "save_strategy": training.save_strategy,
        "save_only_model": training.save_only_model,
        "save_total_limit": training.save_total_limit,
        "fp16": training.fp16,
        "bf16": training.bf16,
        "gradient_checkpointing": training.gradient_checkpointing,
        "dataloader_num_workers": training.dataloader_num_workers,
        "report_to": training.report_to,
        "logging_strategy": "steps",
        "prediction_loss_only": True,
    }
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kwargs[strategy_key] = "steps"
    if training.num_train_epochs is not None:
        kwargs["num_train_epochs"] = training.num_train_epochs
        kwargs.pop("max_steps")
    accepted = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return TrainingArguments(**kwargs)


def build_trainer(**kwargs) -> Trainer:
    accepted = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in kwargs and "tokenizer" not in accepted:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return Trainer(**{key: value for key, value in kwargs.items() if key in accepted})


def main() -> None:
    args = parse_config_args("Train Longformer-GCN on masked language modeling.")
    config = load_config_with_overrides(args)
    prepare_reproducibility(config.seed)
    save_experiment_config(config, config.output_dir)

    tokenizer = load_tokenizer(config.model_name_or_path)
    raw_datasets = load_mlm_dataset(config.dataset)
    tokenized = tokenize_and_group(raw_datasets, tokenizer, config.dataset)
    collator = build_mlm_collator(tokenizer, config.training.mlm_probability, config.dataset.global_attention_on_cls)

    model = build_graph_bert_for_mlm(config.model_name_or_path, config.graph)
    training_args = build_training_args(config)

    trainer = build_trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"] if config.training.do_train else None,
        eval_dataset=tokenized["validation"] if config.training.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[GraphStatsCallback()],
    )

    if config.training.do_train:
        last_checkpoint = get_last_checkpoint(config.output_dir)
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        tokenizer.save_pretrained(config.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    if config.training.do_eval:
        metrics = trainer.evaluate()
        add_perplexity(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
