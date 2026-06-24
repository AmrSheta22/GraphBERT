from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoConfig, AutoModelForMaskedLM

from graphbert.data import load_mlm_dataset, load_tokenizer
from graphbert.utils import load_config_with_overrides, parse_config_args


def main() -> None:
    args = parse_config_args("Download the base Longformer checkpoint, tokenizer, and MLM dataset cache.")
    config = load_config_with_overrides(args)

    AutoConfig.from_pretrained(config.model_name_or_path)
    load_tokenizer(config.model_name_or_path)
    AutoModelForMaskedLM.from_pretrained(config.model_name_or_path)
    load_mlm_dataset(config.dataset)

    print(f"Downloaded model/tokenizer: {config.model_name_or_path}")
    print(f"Downloaded dataset: {config.dataset.name}/{config.dataset.config_name}")


if __name__ == "__main__":
    main()
