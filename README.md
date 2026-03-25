# SASRec — Self-Attentive Sequential Recommendation

Clean PyTorch 3.10 re-implementation of
[SASRec (Kang & McAuley, ICDM 2018)](https://arxiv.org/abs/1808.09781),
designed for academic reproducibility, learning clarity, and engineering extensibility.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment (Python 3.10 required)
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install the package (core dependencies)
pip install -e .

# Optional: enable wandb logging
pip install -e ".[logging]"

# 3. Preprocess a dataset
python scripts/preprocess.py --dataset beauty

# 4. Train and evaluate
python scripts/run_experiment.py --dataset beauty
```

---

## Project Structure

```
sasrec/
├── configs/
│   ├── base.yaml               # Default hyperparameters
│   └── datasets/               # Per-dataset YAML overrides
│       ├── beauty.yaml
│       ├── video_games.yaml
│       ├── steam.yaml
│       └── ml-1m.yaml
├── sasrec/
│   ├── config.py               # OmegaConf config loader
│   ├── utils.py                # set_seed, worker_init_fn
│   ├── data/
│   │   ├── data_info.py        # Dataset registry and download info
│   │   ├── preprocessor.py     # Raw → train/valid/test splits
│   │   ├── sampler.py          # Fixed evaluation negatives
│   │   └── dataset.py          # PyTorch Dataset classes
│   ├── model/
│   │   ├── modules.py          # Multi-head attention + FFN (Pre-LN)
│   │   └── sasrec.py           # SASRec model
│   ├── evaluation/
│   │   └── metrics.py          # NDCG@K and HR@K
│   └── trainer/
│       └── trainer.py          # Training loop, checkpointing, early stopping
└── scripts/
    ├── preprocess.py           # Preprocessing CLI
    ├── run_experiment.py       # Training + evaluation CLI
    └── show_data_info.py       # Dataset info CLI
```

---

## Data Preparation

### Supported Datasets

```bash
python scripts/show_data_info.py
```

| Dataset      | Key           | Notes                              |
|--------------|---------------|------------------------------------|
| Amazon Beauty | `beauty`     | Amazon product reviews             |
| Amazon Video Games | `video_games` | Amazon product reviews        |
| Steam        | `steam`       | Steam gaming reviews (JSON)        |
| MovieLens-1M | `ml-1m`       | Classic CF benchmark               |

### Preprocessing a Built-in Dataset

```bash
# Downloads and preprocesses the dataset
python scripts/preprocess.py --dataset beauty

# Override minimum interaction threshold
python scripts/preprocess.py --dataset beauty --min_interactions 10
```

### Preprocessing a Custom Dataset

```bash
# CSV format: user_id,item_id,timestamp (with header)
python scripts/preprocess.py --input_path /path/to/data.csv --fmt csv --output_dir data/processed/my_data

# Amazon ratings CSV (no header: user,item,rating,timestamp)
python scripts/preprocess.py --input_path ratings.csv --fmt amazon_csv --output_dir data/processed/amazon

# MovieLens .dat or .zip
python scripts/preprocess.py --input_path ml-1m.zip --fmt movielens --output_dir data/processed/ml-1m

# Steam JSON (gzipped or plain)
python scripts/preprocess.py --input_path reviews.json.gz --fmt steam_json --output_dir data/processed/steam
```

---

## Training

### Using a Built-in Dataset Config

```bash
python scripts/run_experiment.py --dataset beauty
```

### Using a Custom Data Directory

```bash
python scripts/run_experiment.py --data_dir data/processed/my_data
```

### Overriding Hyperparameters

Pass OmegaConf-style `key=value` overrides as positional arguments:

```bash
python scripts/run_experiment.py --dataset beauty \
    model.hidden_units=128 \
    train.lr=0.0005 \
    train.num_epochs=300 \
    data.maxlen=200
```

### Key Config Options (`configs/base.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `model.hidden_units` | 50 | Embedding and hidden dimension |
| `model.num_blocks` | 2 | Number of Transformer blocks |
| `model.num_heads` | 1 | Number of attention heads |
| `model.dropout_rate` | 0.2 | Dropout probability |
| `data.maxlen` | 50 | Maximum sequence length |
| `data.num_neg_eval` | 100 | Sampled negatives per user at eval |
| `train.lr` | 0.001 | Adam learning rate |
| `train.batch_size` | 128 | Training batch size |
| `train.num_epochs` | 200 | Maximum training epochs |
| `train.patience` | 20 | Early-stopping patience (epochs) |
| `eval.k_values` | [5, 10, 20] | NDCG@K and HR@K cutoffs |
| `logging.use_wandb` | false | Enable wandb logging |

---

## Evaluation Protocol

- **Leave-one-out split**: test = last item, valid = second-to-last, train = rest.
- **Sampled evaluation** (default): 100 fixed random negatives + 1 positive = 101 candidates per user.
- **Primary metric**: NDCG@10 for early stopping and best checkpoint selection.
- **Test evaluation**: uses the best checkpoint (by val NDCG@10); validation item is included as context.

---

## Logging

### TensorBoard (always on)

```bash
tensorboard --logdir runs/
```

### wandb (optional)

```bash
pip install -e ".[logging]"
python scripts/run_experiment.py --dataset beauty logging.use_wandb=true logging.wandb_project=my_project
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Architecture Notes

- **Pre-LN Transformer**: LayerNorm applied before each sublayer (matching the original SASRec codebase).
- **ReLU activation** in the pointwise FFN (not GELU).
- **Causal attention mask**: future tokens are masked; attending to padding is also masked.
- **Weight tying**: input item embeddings are shared with the output scoring layer.
- **1-indexed positional embeddings**: position 0 is padding; positions 1..maxlen map to sequence slots left-to-right.
- **Training loss**: per-position BCE with `istarget = (pos != 0).float()` to mask padding.

---

## Citation

```bibtex
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```
