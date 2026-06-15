# GraphBERT Attention-GCN Prototype

Research prototype for testing whether explicit graph message passing over a Transformer's learned attention graph improves masked language modeling.

The model starts from Hugging Face `bert-large-uncased`. Selected final encoder layers replace vanilla self-attention internals with:

```text
Multi-head self-attention
  -> sparsify each head's attention matrix
  -> treat attention as a weighted token graph
  -> per-head GCN propagation over that head's contextual output
  -> concatenate heads
  -> standard BERT output projection, residuals, layer norms, and FFN
```

This is not a parallel GNN branch. The attention probabilities themselves are the dynamic graph.

## Project Layout

```text
configs/
  graphbert_wikitext103.yaml
scripts/
  train_mlm.py
  evaluate_mlm.py
  download_assets.py
graphbert/
  config.py
  data.py
  graph_attention.py
  metrics.py
  modeling.py
  utils.py
requirements.txt
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Download the base checkpoint and dataset cache without starting training:

```bash
python scripts/download_assets.py --config configs/graphbert_wikitext103.yaml
```

## Training

The default config uses WikiText-103 for a lightweight public masked-language-modeling demonstration. BookCorpus + Wikipedia can be substituted in `configs/graphbert_wikitext103.yaml` when a full pretraining-scale setup is desired.

```bash
accelerate launch scripts/train_mlm.py --config configs/graphbert_wikitext103.yaml
```

Useful overrides:

```bash
python scripts/train_mlm.py \
  --config configs/graphbert_wikitext103.yaml \
  --num-replaced-layers 2 \
  --sparsification topk \
  --top-k 16
```

Set `num_replaced_layers: 0` for a vanilla BERT baseline. Avoid replacing all layers unless you intentionally want an unstable ablation.

## Evaluation

```bash
python scripts/evaluate_mlm.py \
  --config configs/graphbert_wikitext103.yaml \
  --checkpoint outputs/graphbert/checkpoint-1000
```

Evaluation reports MLM loss, perplexity, average graph sparsity, average node degree, and surviving-edge percentage.

## Key Configuration Options

- `model_name_or_path`: defaults to `bert-large-uncased`.
- `num_replaced_layers`: number of final encoder layers to replace.
- `sparsification`: `dense`, `threshold`, or `topk`.
- `threshold`: attention edge cutoff for threshold sparsification.
- `top_k`: number of outgoing neighbors per token for top-k sparsification.
- `renormalize_adjacency`: row-normalize after sparsification.
- `symmetric_normalization`: apply GCN-style `D^-1/2 A D^-1/2` normalization.

## Implementation Notes

- Replacement happens after `BertForMaskedLM.from_pretrained(...)`, preserving standard BERT parameters wherever possible.
- Query/key/value and output/FFN modules remain checkpoint-compatible.
- Each replaced self-attention module adds only a per-head GCN weight tensor initialized to identity.
- Graph statistics are collected from replaced layers during forward passes and logged by a Hugging Face `TrainerCallback`.
- The code favors clarity over fused kernels. The attention graph is kept dense after sparsification masks, which is simple and suitable for sequence-length research experiments.
