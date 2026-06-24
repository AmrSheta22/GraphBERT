# Longformer-GCN Sparse Context Prototype

Research prototype for testing whether graph convolution over Longformer's sparse attention topology improves masked language modeling on long sequences.

The base model is Hugging Face `allenai/longformer-base-4096`. Standard Longformer layers use sliding-window self-attention, with optional global-attention tokens, so the model never constructs dense all-token attention. Configured encoder layers are replaced by GCN blocks:

```text
Longformer encoder
  -> local sliding-window attention layers
  -> selected GCN replacement layers
       adjacency = Longformer local-window edges + global-token edges
       aggregation = normalized neighborhood sum
       projection = trainable GCN linear layer
       output = pretrained residual/layer norm + pretrained FFN
  -> masked-language-modeling head
```

The GCN path does not materialize an `n x n` adjacency matrix. Local aggregation uses cumulative sliding sums, and global-token aggregation uses reductions over the sequence. This preserves the sparse long-context objective.

## Project layout

```text
configs/
  graphbert_wikitext103.yaml
scripts/
  train_mlm.py
  evaluate_mlm.py
  download_assets.py
  run_sequential_search.sh
graphbert/
  config.py
  data.py
  graph_attention.py
  metrics.py
  modeling.py
  utils.py
```

## Setup

```bash
python -m pip install -r requirements.txt
python scripts/download_assets.py --config configs/graphbert_wikitext103.yaml
```

## Training

The default experiment uses 4096-token WikiText-103 blocks, one global CLS token, and replaces the final two Longformer layers.

```bash
accelerate launch scripts/train_mlm.py \
  --config configs/graphbert_wikitext103.yaml
```

Useful overrides:

```bash
# Replace the final four layers
python scripts/train_mlm.py \
  --config configs/graphbert_wikitext103.yaml \
  --num-replaced-layers 4 \
  --replacement-strategy final

# Replace two intermediate layers explicitly (zero-based indices)
python scripts/train_mlm.py \
  --config configs/graphbert_wikitext103.yaml \
  --layer-indices 5 8

# Spread three GCN blocks through the encoder
python scripts/train_mlm.py \
  --config configs/graphbert_wikitext103.yaml \
  --num-replaced-layers 3 \
  --replacement-strategy uniform
```

Set `num_replaced_layers: 0` for the vanilla Longformer baseline.

## Layer placement

- `final`: replace the last `num_replaced_layers` blocks.
- `intermediate`: replace a centered contiguous group of blocks.
- `first`: replace the first blocks.
- `uniform`: distribute replacements across encoder depth; one replacement selects the middle layer.
- `explicit`: use the zero-based `layer_indices` list. Its length must match `num_replaced_layers`.

## Sparse graph behavior

For a configured Longformer attention window of width `w`:

- ordinary tokens aggregate non-padding neighbors in their local window;
- global-attention tokens aggregate every non-padding token;
- ordinary tokens also aggregate all global-attention tokens;
- padding nodes neither send nor receive messages;
- self edges, row normalization, symmetric normalization, activation, dropout, and GCN initialization are configurable.

The default collator marks the first token (CLS) for global attention. Disable this with `dataset.global_attention_on_cls: false`.

## Evaluation

```bash
python scripts/evaluate_mlm.py \
  --config configs/graphbert_wikitext103.yaml \
  --checkpoint outputs/longformer-gcn-wikitext103/checkpoint-1000
```

Evaluation reports MLM loss, perplexity, average graph degree, edge count, and valid-node count for replaced layers.

## Sequential experiment search

```bash
bash scripts/run_sequential_search.sh
```

The search compares a vanilla Longformer baseline with final, uniformly distributed, and explicitly placed GCN replacements. At 4096 tokens, start with batch size 1 and gradient accumulation; tune those values for your hardware.

## Implementation notes

- Replacement happens after `LongformerForMaskedLM.from_pretrained(...)`.
- A replacement removes that block's learned self-attention, but reuses its pretrained attention-output residual/layer norm and feed-forward modules.
- The GCN projection is initialized to identity by default so initial behavior begins as normalized neighborhood smoothing followed by the pretrained remainder of the block.
- Non-replaced layers retain native Longformer local and global attention.
- Replaced layers use Longformer's graph topology, not its learned attention probabilities.
## Long-context retrieval evaluation (MLDR)

This repository includes an English MLDR evaluation modeled on ModernBERT Section 3.1.3. MLDR contains 10,000 English training queries, 800 test queries, and a 200,000-document corpus. Results are reported as nDCG@10; Recall@100 is also written as a diagnostic.

The adaptation uses Longformer's native 4096-token limit rather than ModernBERT's 8192-token limit. Queries default to 64 tokens and MLDR documents to 4096 tokens. Every document is encoded as one long sequence—documents are not passage-chunked—so the benchmark measures long-document representations.

### Install retrieval dependencies

The single-vector path uses the standard project requirements. Multi-vector indexing additionally requires PyLate:

```bash
pip install pylate
```

### 1. Single Vector — Out of Domain

Train on 1.25M MS MARCO examples with mined hard negatives, batch size 16, and 5% warmup, then evaluate directly on the MLDR test corpus:

```bash
python scripts/train_retrieval.py \
  --config configs/graphbert_wikitext103.yaml \
  --source-model outputs/longformer-gcn-wikitext103/checkpoint-1000 \
  --output-dir outputs/mldr/single-msmarco \
  --stage msmarco \
  --architecture single \
  --batch-size 16 \
  --warmup-ratio 0.05 \
  --max-samples 1250000 \
  --fp16

python scripts/evaluate_mldr.py \
  --checkpoint outputs/mldr/single-msmarco \
  --output-dir outputs/mldr/single-ood \
  --document-max-length 4096
```

The evaluator performs exact cosine-similarity retrieval over all 200,000 documents and writes `metrics.json` plus a TREC run file.

### 2. Single Vector — In Domain

Continue the MS MARCO checkpoint on the 10,000-query MLDR English training split, then reevaluate:

```bash
python scripts/train_retrieval.py \
  --config configs/graphbert_wikitext103.yaml \
  --source-model outputs/mldr/single-msmarco \
  --output-dir outputs/mldr/single-mldr \
  --stage mldr \
  --architecture single \
  --batch-size 16 \
  --warmup-ratio 0.05 \
  --max-samples 10000 \
  --document-max-length 4096 \
  --fp16

python scripts/evaluate_mldr.py \
  --checkpoint outputs/mldr/single-mldr \
  --output-dir outputs/mldr/single-id
```

### 3. Multi Vector — Out of Domain

The ColBERT path follows the paper's 809k-query MS MARCO setup: 32 candidate documents per query, BGE-M3 teacher scores, KL-divergence distillation, batch size 16, and 5% warmup. It is evaluated on MLDR without MLDR fine-tuning. Token embeddings are indexed with PyLate's PLAID index and scored using MaxSim.

```bash
python scripts/train_retrieval.py \
  --config configs/graphbert_wikitext103.yaml \
  --source-model outputs/longformer-gcn-wikitext103/checkpoint-1000 \
  --output-dir outputs/mldr/colbert-msmarco \
  --stage msmarco \
  --architecture colbert \
  --projection-dim 128 \
  --colbert-candidates 32 \
  --batch-size 16 \
  --warmup-ratio 0.05 \
  --max-samples 810000 \
  --fp16

python scripts/evaluate_mldr_colbert.py \
  --checkpoint outputs/mldr/colbert-msmarco \
  --output-dir outputs/mldr/colbert-ood \
  --index-folder outputs/mldr/colbert-index \
  --document-max-length 4096
```

Run all three settings sequentially with:

```bash
SOURCE_MODEL=outputs/longformer-gcn-wikitext103/checkpoint-1000 \
OUTPUT_ROOT=outputs/mldr \
bash scripts/run_mldr_evaluation.sh
```

For paper-style model selection, repeat MS MARCO training over a learning-rate sweep and select the checkpoint using a fixed BEIR development subset before running MLDR. Do not select learning rates on the MLDR test split.

For inexpensive smoke tests, both evaluators accept `--max-corpus-documents`; results from a truncated corpus are not benchmark scores.
