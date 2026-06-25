from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphbert.data import build_mlm_collator, load_mlm_dataset, load_tokenizer, tokenize_and_group
from graphbert.metrics import GraphStatsCallback, add_perplexity, collect_graph_stats
from graphbert.modeling import load_graph_bert_checkpoint
from graphbert.utils import compact_metrics, load_config_with_overrides, parse_config_args, prepare_reproducibility
from scripts.train_mlm import build_trainer


def main() -> None:
    args = parse_config_args("Evaluate a Longformer-APPNP MLM checkpoint.")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluation")

    config = load_config_with_overrides(args)
    prepare_reproducibility(config.seed)

    tokenizer = load_tokenizer(args.checkpoint)
    raw_datasets = load_mlm_dataset(config.dataset)
    tokenized = tokenize_and_group(raw_datasets, tokenizer, config.dataset)
    collator = build_mlm_collator(tokenizer, config.training.mlm_probability, config.dataset.global_attention_on_cls)

    model = load_graph_bert_checkpoint(args.checkpoint, config.graph)

    from scripts.train_mlm import build_training_args
    training_args = build_training_args(config)
    training_args.do_train = False

    trainer = build_trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[GraphStatsCallback()],
    )

    metrics = trainer.evaluate()
    metrics.update(collect_graph_stats(model))
    add_perplexity(metrics)
    metrics = compact_metrics(metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
