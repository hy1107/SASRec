# SASRec Python 3.10 Rewrite — Design Spec

**Date:** 2026-03-25
**Framework:** PyTorch
**Python:** 3.10
**Source reference:** [kang205/SASRec](https://github.com/kang205/SASRec) (ICDM 2018)

---

## Goals

1. **Primary — Academic research / paper reproduction:** Faithfully replicate the SASRec training setup, evaluation protocol (leave-one-out, NDCG@10, HR@10), and reported results on Amazon and MovieLens datasets.
2. **Secondary — Clarity for learning:** Clean, well-commented code with each module having a single clear purpose.
3. **Secondary — Engineering extensibility:** Modular design enabling new features, custom datasets, and experiment variants without restructuring.

---

## Project Structure

```
sasrec/
├── configs/
│   ├── base.yaml                  # Default hyperparameters
│   └── datasets/
│       ├── beauty.yaml
│       ├── video_games.yaml
│       ├── steam.yaml
│       └── ml-1m.yaml
│
├── data/
│   ├── raw/                       # Downloaded raw data (gitignored)
│   └── processed/                 # Preprocessed data (gitignored)
│
├── sasrec/                        # Main package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_info.py           # Print download URLs and instructions
│   │   ├── preprocessor.py        # Raw → standard format
│   │   ├── dataset.py             # PyTorch Dataset / DataLoader
│   │   └── sampler.py             # Negative sampling
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── sasrec.py              # Main model
│   │   └── modules.py             # MultiHeadAttention, FFN, LayerNorm
│   │
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── trainer.py             # Training loop, checkpoint, early stopping
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py             # NDCG@K, HR@K, sampled & full-ranking
│
├── scripts/
│   ├── show_data_info.py          # Print dataset download instructions
│   ├── preprocess.py              # Data preprocessing
│   └── run_experiment.py          # Main training entry point
│
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Architecture

### Model (`sasrec/model/`)

**`modules.py`** — Reusable Transformer primitives:
- `LayerNorm`: standard layer normalization
- `MultiHeadAttention`: causal (lower-triangular) masked self-attention; padding key positions detected by checking whether the input token id is 0 (not via zero-vector heuristic — using token ids is unambiguous and unaffected by weight magnitude); **Pre-LN style** (LayerNorm applied before each sublayer, matching the original SASRec codebase)
- `PointwiseFeedForward`: two Linear layers with **ReLU** activation and dropout (matching original SASRec codebase; GELU would be a deviation from Goal 1)

**`sasrec.py`** — Main model:
- Item embedding table of shape `[item_num+1, hidden_units]`, index 0 reserved for padding (gradient zeroed via `padding_idx=0`)
- Learned positional embedding table of shape `[maxlen+1, hidden_units]` with `padding_idx=0`; position indices are **1-indexed** and assigned to slots in the fixed-length padded window (not derived from the user's original interaction timeline): index 1 = leftmost (oldest) slot, index `maxlen` = rightmost (most recent) slot; padding slots (token id = 0) are assigned position index 0, which is suppressed by `padding_idx=0` so padding positions never receive a positional embedding gradient
- Weight-tying: same item embedding table used for output prediction (dot product)
- N stacked Transformer blocks: **Pre-LN → CausalMHA → Residual → Pre-LN → FFN → Residual → Dropout** (Pre-LN, matching original codebase)
- Output: last non-padding hidden state for next-item prediction

### Data (`sasrec/data/`)

**`data_info.py`**: Library module containing download URLs and expected filenames for each built-in dataset. `scripts/show_data_info.py` calls this module and prints human-readable instructions. Does not attempt automatic downloads (URLs change over time and network-dependent tests are brittle). Users download manually and place files in `data/raw/`.

**`preprocessor.py`**:
- Parses raw Amazon JSON / MovieLens dat files, and generic CSV files
- **CSV format**: must have a header row with columns named exactly `user_id`, `item_id`, `timestamp` (comma-delimited); `timestamp` must be a numeric value sortable in ascending order; other columns are ignored
- Filters users/items with fewer than `min_interactions` (default 5)
- Remaps user/item IDs to contiguous integers starting from 1
- Sorts interactions by timestamp
- Outputs: `{split}.txt` files in format `user_id item_1 item_2 ... item_n`, plus `item_count.txt` with total number of unique items

**`dataset.py`**:
- Training: for each user, uses the full interaction sequence **excluding the last 2 items** (val and test), truncated to `maxlen` from the right. For a user with T total interactions, the training sequence has T-2 items. The input tensor `seq` has shape `[maxlen]` (left-padded with 0); the positive target tensor `pos` = `seq` shifted left by 1 and padded back to `[maxlen]` (i.e., `pos[i] = seq[i+1]` for `i < maxlen-1`, `pos[maxlen-1]` = the (T-2)-th item); `neg` is a same-shape tensor of sampled negatives. All non-padding positions contribute to the loss. This matches the original SASRec training exactly — no sliding window is generated.
- **Edge case for short sequences**: users with `min_interactions=5` will have 3 training items after removing val and test. This produces a sequence with 3 non-padding positions and 47 padding positions (for `maxlen=50`) — this is valid and included in training without special handling.
- Validation/Test: leave-one-out; val uses second-to-last item, test uses last item; input is the sequence truncated to `maxlen` excluding the target item
- All sequences are left-padded to `maxlen` with 0

**`sampler.py`**:
- Training: dynamic uniform negative sampling per batch (rejection sampling, avoids user's full history)
- Evaluation: two independent fixed negative sets drawn at startup — one for val, one for test — each seeded separately. Each set contains `num_neg_eval` (default 100) negatives per user, excluding the user's full interaction history (not just training split). Reused across all eval epochs.

### Trainer (`sasrec/trainer/`)

**`trainer.py`**:
- Optimizer: Adam with configurable lr, beta1, beta2, weight decay
- Loss: Binary Cross-Entropy summed over all non-padding positions in the sequence. For each position `t`, the model predicts `pos[t]` (the positive target item at that slot) against `neg[t]` (a sampled negative). `istarget` is a binary mask of shape `[batch, maxlen]` computed as `(pos != 0).float()`, where `pos` is the positive target tensor precomputed in `dataset.py` as the input sequence shifted left by 1 slot (see `dataset.py` description above); `istarget` is 1 wherever `pos` is non-zero, ensuring padding positions contribute zero gradient.
- Checkpoint: saves best model by NDCG@10 on validation set (`best.pt`) and overwrites the latest checkpoint (`latest.pt`) each epoch; keeps only these two files, no rotation
- Early stopping: stops if validation NDCG@10 does not improve for `patience` epochs
- Logging: TensorBoard always; wandb optional (enabled via config flag)
- Reproducibility: global random seed set at start; fixed evaluation negatives

### Evaluation (`sasrec/evaluation/`)

**`metrics.py`**:
- `ndcg_at_k(scores, labels, k)` — normalized discounted cumulative gain
- `hit_at_k(scores, labels, k)` — hit rate
- Mode `sampled`: ranks 1 positive against 100 fixed negatives (101 candidates total); fast, matches original paper protocol
- Mode `full`: ranks the target item (positive) against all catalog items **not in the user's interaction history except the target item itself** — i.e., candidate set = {target item} ∪ {items the user has never interacted with}; the target item is always included as the positive candidate; items the user interacted with other than the target are excluded to avoid penalizing correct predictions; expensive but unbiased
- K is configurable (default evaluates @5, @10, @20 simultaneously)

---

## Configuration

`configs/base.yaml` holds all defaults. Dataset yamls override only the fields that differ.

```yaml
# configs/base.yaml
data:
  maxlen: 50
  min_interactions: 5
  num_neg_train: 1
  num_neg_eval: 100

model:
  hidden_units: 50
  num_blocks: 2
  num_heads: 1
  dropout_rate: 0.5

train:
  batch_size: 128
  lr: 0.001
  num_epochs: 200
  patience: 20
  seed: 42

eval:
  k_values: [5, 10, 20]
  eval_mode: sampled   # or "full"

logging:
  use_wandb: false
  wandb_project: "sasrec"
  log_dir: "runs/"
  checkpoint_dir: "checkpoints/"
```

---

## Dependencies

Config management uses **OmegaConf** for YAML loading and dot-notation CLI overrides. The CLI wrapper in `run_experiment.py` accepts `--key.subkey value` arguments and merges them into the base config via `OmegaConf.merge`.

Minimum `requirements.txt`:
```
torch>=2.0
numpy>=1.24
omegaconf>=2.3
tensorboard>=2.13
tqdm>=4.65
```

`pyproject.toml` (standard Python 3.10 packaging); `wandb` is optional to avoid forcing installation in offline/HPC environments:
```toml
[build-system]
requires = ["setuptools>=67"]
build-backend = "setuptools.build_meta"

[project]
name = "sasrec"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch>=2.0", "numpy>=1.24", "omegaconf>=2.3", "tensorboard>=2.13", "tqdm>=4.65"]

[project.optional-dependencies]
logging = ["wandb>=0.15"]
```

Install with wandb: `pip install -e ".[logging]"`. The `wandb` package is imported at runtime only when `use_wandb: true`; if wandb is not installed and `use_wandb: true` is set, a clear `ImportError` is raised with instructions to install the `logging` extra.

---

## Data Flow

```
Raw data (Amazon JSON / MovieLens dat)
    ↓ preprocessor.py
Standard format: user_id → [item_1, item_2, ..., item_n] (time-sorted)
Saved as train/valid/test splits + item count metadata
    ↓ dataset.py
Training:  full sequence per user (all non-padding positions) + dynamic negative sampling
Val/Test:  leave-one-out + fixed 100 sampled negatives (or full catalog minus user history)
    ↓ DataLoader (multi-worker)
Trainer: forward pass → BCE loss → backprop → metrics every N epochs
    ↓ metrics.py
NDCG@K, HR@K reported; best checkpoint saved; early stop if no improvement
```

---

## Error Handling

- `data_info.py`: prints download instructions; `preprocessor.py` raises `FileNotFoundError` with clear message if expected raw files are missing when preprocessing is attempted
- `preprocessor.py`: warns (not crashes) if a dataset has unusually few users/items after filtering
- `trainer.py`: catches exceptions with `logging.exception()` to preserve full traceback before re-raising
- No bare `except` clauses anywhere

---

## Reproducibility

- Global seed set once at startup: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `numpy.random.seed(seed)`, `random.seed(seed)`
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` when seed is set
- `torch.use_deterministic_algorithms(True)` enabled for full PyTorch 2.x determinism guarantee; note that full GPU determinism is best-effort and may not hold across different hardware or CUDA versions
- DataLoader workers seeded via `worker_init_fn` derived from the global seed
- Evaluation negative sets generated once at startup (before training begins) and reused across all eval epochs; val set uses `seed` as its random seed, test set uses `seed + 1`; this derivation is fixed so changing the global seed shifts both sets predictably

---

## Usage (Quick Reference)

```bash
# 1. Install
pip install -e .

# 2. Get data download instructions
python scripts/show_data_info.py --dataset beauty
# Then manually download and place in data/raw/

# 3. Preprocess
python scripts/preprocess.py --dataset beauty

# 4. Train (uses configs/datasets/beauty.yaml + configs/base.yaml)
python scripts/run_experiment.py --dataset beauty

# 5. Custom dataset (CSV with user_id, item_id, timestamp columns)
#    --dataset and --input are mutually exclusive; --dataset takes precedence if both given (error raised)
#    Output is saved to data/processed/<stem of input filename>/
python scripts/preprocess.py --input my_data.csv --format csv
python scripts/run_experiment.py --data_dir data/processed/my_data

# 6. Override hyperparameters from CLI
python scripts/run_experiment.py --dataset beauty --model.hidden_units 64 --train.lr 0.0005

# 7. Enable wandb
python scripts/run_experiment.py --dataset beauty --logging.use_wandb true
```

---

## Key Improvements Over Original

| Issue in original | Fix in this rewrite |
|---|---|
| Python 2 only | Python 3.10 throughout |
| TF 1.x only | PyTorch |
| No checkpoint saving | `trainer.py` saves `best.pt` and `latest.pt` |
| Bare `except` swallows errors | `logging.exception()` + re-raise |
| Test candidate count hardcoded to 101 | Configurable via `num_neg_eval` |
| Dead `positional_encoding` function | Removed |
| Wrong scope name in `feedforward` | N/A (PyTorch modules are class-based) |
| No reproducibility guarantee | Fixed seed + deterministic eval negatives |
| No wandb / TensorBoard | Both supported |
| No config file | YAML config with CLI overrides |
