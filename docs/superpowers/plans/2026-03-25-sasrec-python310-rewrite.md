# SASRec Python 3.10 Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the SASRec sequential recommendation model from Python 2 / TensorFlow 1.x to Python 3.10 / PyTorch with clean modular architecture, faithful paper reproduction, and full training management.

**Architecture:** Four-layer package (`data`, `model`, `trainer`, `evaluation`) under `sasrec/`, each with a single responsibility. OmegaConf YAML configs drive all hyperparameters; scripts in `scripts/` are thin entry points. TDD throughout — write the failing test first, then the implementation.

**Tech Stack:** Python 3.10, PyTorch ≥ 2.0, OmegaConf ≥ 2.3, TensorBoard ≥ 2.13, tqdm, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies |
| `requirements.txt` | Pinned dev dependencies |
| `configs/base.yaml` | Default hyperparameters |
| `configs/datasets/*.yaml` | Per-dataset overrides |
| `sasrec/__init__.py` | Package root |
| `sasrec/data/__init__.py` | Data subpackage root |
| `sasrec/data/data_info.py` | Dataset URLs and expected filenames |
| `sasrec/data/preprocessor.py` | Raw → standard split format |
| `sasrec/data/dataset.py` | PyTorch Dataset for train/val/test |
| `sasrec/data/sampler.py` | Negative sampling (train dynamic, eval fixed) |
| `sasrec/model/__init__.py` | Model subpackage root |
| `sasrec/model/modules.py` | LayerNorm, MultiHeadAttention, PointwiseFeedForward |
| `sasrec/model/sasrec.py` | SASRec main model |
| `sasrec/trainer/__init__.py` | Trainer subpackage root |
| `sasrec/trainer/trainer.py` | Training loop, checkpoint, early stopping |
| `sasrec/utils.py` | Seed helper, worker_init_fn |
| `sasrec/evaluation/__init__.py` | Evaluation subpackage root |
| `sasrec/evaluation/metrics.py` | NDCG@K, HR@K (sampled and full-ranking) |
| `scripts/show_data_info.py` | CLI: print dataset download instructions |
| `scripts/preprocess.py` | CLI: run preprocessor |
| `scripts/run_experiment.py` | CLI: train + evaluate |
| `tests/test_preprocessor.py` | Preprocessor unit tests |
| `tests/test_dataset.py` | Dataset unit tests |
| `tests/test_sampler.py` | Sampler unit tests |
| `tests/test_modules.py` | Transformer module unit tests |
| `tests/test_sasrec.py` | Full model unit tests |
| `tests/test_metrics.py` | Metric function unit tests |
| `tests/test_trainer.py` | Trainer integration test |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `sasrec/__init__.py`
- Create: `sasrec/data/__init__.py`
- Create: `sasrec/model/__init__.py`
- Create: `sasrec/trainer/__init__.py`
- Create: `sasrec/evaluation/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=67"]
build-backend = "setuptools.build_meta"

[project]
name = "sasrec"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "omegaconf>=2.3",
    "tensorboard>=2.13",
    "tqdm>=4.65",
]

[project.optional-dependencies]
logging = ["wandb>=0.15"]
dev = ["pytest>=7.0", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["."]
include = ["sasrec*"]
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.0
numpy>=1.24
omegaconf>=2.3
tensorboard>=2.13
tqdm>=4.65
pytest>=7.0
pytest-cov
```

- [ ] **Step 3: Create package `__init__.py` files**

Each file is empty (just a marker). Create:
- `sasrec/__init__.py`
- `sasrec/data/__init__.py`
- `sasrec/model/__init__.py`
- `sasrec/trainer/__init__.py`
- `sasrec/evaluation/__init__.py`
- `tests/__init__.py`

- [ ] **Step 4: Create data directories and .gitignore**

```bash
mkdir -p data/raw data/processed
touch data/raw/.gitkeep data/processed/.gitkeep
```

`.gitignore`:
```
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
__pycache__/
*.pyc
*.egg-info/
runs/
checkpoints/
.env
wandb/
```

- [ ] **Step 5: Install package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors; `python -c "import sasrec"` succeeds silently.

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml requirements.txt sasrec/ tests/ data/ .gitignore
git commit -m "chore: project scaffolding"
```

---

## Task 2: Configuration System

**Files:**
- Create: `configs/base.yaml`
- Test: `tests/test_config.py`
- Create: `configs/datasets/beauty.yaml`
- Create: `configs/datasets/video_games.yaml`
- Create: `configs/datasets/steam.yaml`
- Create: `configs/datasets/ml-1m.yaml`
- Create: `sasrec/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
from sasrec.config import load_config

def test_load_base_config():
    cfg = load_config()
    assert cfg.data.maxlen == 50
    assert cfg.model.hidden_units == 50
    assert cfg.train.seed == 42

def test_dataset_override():
    cfg = load_config(dataset="beauty")
    assert cfg.data.maxlen == 50  # beauty uses base default

def test_cli_override():
    cfg = load_config(overrides=["model.hidden_units=64"])
    assert cfg.model.hidden_units == 64

def test_unknown_dataset_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_config(dataset="nonexistent")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```
Expected: `ImportError: cannot import name 'load_config'`

- [ ] **Step 3: Create configs/base.yaml**

```yaml
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
  beta1: 0.9
  beta2: 0.98
  weight_decay: 0.0
  num_epochs: 200
  patience: 20
  seed: 42

eval:
  k_values: [5, 10, 20]
  eval_mode: sampled

logging:
  use_wandb: false
  wandb_project: "sasrec"
  log_dir: "runs/"
  checkpoint_dir: "checkpoints/"
```

- [ ] **Step 4: Create dataset config files**

`configs/datasets/beauty.yaml`:
```yaml
# Amazon Beauty — matches paper settings
data:
  maxlen: 50
```

`configs/datasets/video_games.yaml`:
```yaml
data:
  maxlen: 50
model:
  dropout_rate: 0.5
```

`configs/datasets/steam.yaml`:
```yaml
data:
  maxlen: 50
model:
  hidden_units: 50
  num_blocks: 2
```

`configs/datasets/ml-1m.yaml`:
```yaml
data:
  maxlen: 200
model:
  hidden_units: 100
  num_blocks: 2
  num_heads: 2
  dropout_rate: 0.2
train:
  batch_size: 256
```

- [ ] **Step 5: Implement sasrec/config.py**

```python
# sasrec/config.py
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(
    dataset: str | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load base config, optionally merge dataset-specific overrides and CLI overrides.

    Args:
        dataset: Name of dataset (e.g. "beauty"). Looks for configs/datasets/{dataset}.yaml.
        overrides: List of OmegaConf dot-notation override strings, e.g. ["model.hidden_units=64"].

    Returns:
        Merged DictConfig.

    Raises:
        FileNotFoundError: If the dataset config file does not exist.
    """
    cfg = OmegaConf.load(_CONFIGS_DIR / "base.yaml")

    if dataset is not None:
        dataset_cfg_path = _CONFIGS_DIR / "datasets" / f"{dataset}.yaml"
        if not dataset_cfg_path.exists():
            raise FileNotFoundError(
                f"No config found for dataset '{dataset}'. "
                f"Expected: {dataset_cfg_path}"
            )
        cfg = OmegaConf.merge(cfg, OmegaConf.load(dataset_cfg_path))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add configs/ sasrec/config.py tests/test_config.py
git commit -m "feat: config system with OmegaConf YAML + CLI overrides"
```

---

## Task 3: Data Info Module

**Files:**
- Create: `sasrec/data/data_info.py`
- Create: `scripts/show_data_info.py`
- Test: `tests/test_data_info.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_data_info.py
from sasrec.data.data_info import get_dataset_info, SUPPORTED_DATASETS

def test_supported_datasets_exist():
    for name in ["beauty", "video_games", "steam", "ml-1m"]:
        assert name in SUPPORTED_DATASETS

def test_get_dataset_info_returns_url_and_filename():
    info = get_dataset_info("beauty")
    assert "url" in info
    assert "raw_filename" in info
    assert isinstance(info["url"], str)
    assert len(info["url"]) > 0

def test_unknown_dataset_raises():
    import pytest
    with pytest.raises(KeyError):
        get_dataset_info("unknown_dataset")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_data_info.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement sasrec/data/data_info.py**

```python
# sasrec/data/data_info.py
"""Dataset download information for built-in SASRec datasets.

Users must download raw files manually and place them in data/raw/.
Run `python scripts/show_data_info.py --dataset <name>` for instructions.
"""

SUPPORTED_DATASETS: dict[str, dict] = {
    "beauty": {
        "url": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv",
        "raw_filename": "ratings_Beauty.csv",
        "format": "amazon_csv",
        "description": "Amazon Beauty product ratings",
    },
    "video_games": {
        "url": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv",
        "raw_filename": "ratings_Video_Games.csv",
        "format": "amazon_csv",
        "description": "Amazon Video Games ratings",
    },
    "steam": {
        "url": "https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
        "raw_filename": "steam_reviews.json.gz",
        "format": "steam_json",
        "description": "Steam game reviews (Wang-Cheng Kang's processed version)",
    },
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "raw_filename": "ml-1m.zip",
        "format": "movielens",
        "description": "MovieLens 1M dataset",
    },
}


def get_dataset_info(name: str) -> dict:
    """Return download info for a built-in dataset.

    Raises:
        KeyError: If `name` is not in SUPPORTED_DATASETS.
    """
    if name not in SUPPORTED_DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Supported: {list(SUPPORTED_DATASETS.keys())}"
        )
    return SUPPORTED_DATASETS[name]


def format_instructions(name: str) -> str:
    """Return human-readable download instructions for a dataset."""
    info = get_dataset_info(name)
    return (
        f"Dataset: {name}\n"
        f"Description: {info['description']}\n"
        f"Download URL: {info['url']}\n"
        f"Save file to: data/raw/{info['raw_filename']}\n"
        f"Then run: python scripts/preprocess.py --dataset {name}\n"
    )
```

- [ ] **Step 4: Create scripts/show_data_info.py**

```python
#!/usr/bin/env python3
"""Print download instructions for built-in SASRec datasets."""
import argparse
from sasrec.data.data_info import format_instructions, SUPPORTED_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Show data download instructions")
    parser.add_argument(
        "--dataset",
        choices=list(SUPPORTED_DATASETS.keys()),
        help="Dataset name. Omit to list all.",
    )
    args = parser.parse_args()

    if args.dataset:
        print(format_instructions(args.dataset))
    else:
        print("Available datasets:\n")
        for name in SUPPORTED_DATASETS:
            print(format_instructions(name))
            print("-" * 40)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_data_info.py -v
```
Expected: 3 passed

- [ ] **Step 6: Smoke-test the script**

```bash
python scripts/show_data_info.py --dataset beauty
```
Expected: prints URL and instructions for Beauty dataset.

- [ ] **Step 7: Commit**

```bash
git add sasrec/data/data_info.py scripts/show_data_info.py tests/test_data_info.py
git commit -m "feat: data info module and show_data_info script"
```

---

## Task 4: Preprocessor

**Files:**
- Create: `sasrec/data/preprocessor.py`
- Create: `scripts/preprocess.py`
- Test: `tests/test_preprocessor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocessor.py
import tempfile
import os
from pathlib import Path
import pytest
from sasrec.data.preprocessor import preprocess, load_processed_data


@pytest.fixture
def sample_csv(tmp_path):
    """Create a tiny CSV interaction file."""
    csv_file = tmp_path / "interactions.csv"
    # 4 users, each with >= 5 interactions; items 1-10
    rows = []
    for uid in range(1, 5):
        for iid in range(1, 8):  # 7 interactions per user
            rows.append(f"{uid},{iid},{uid * 100 + iid}")
    csv_file.write_text("user_id,item_id,timestamp\n" + "\n".join(rows))
    return csv_file


def test_preprocess_csv_creates_output_files(sample_csv, tmp_path):
    out_dir = tmp_path / "processed"
    preprocess(input_path=sample_csv, output_dir=out_dir, fmt="csv", min_interactions=5)
    assert (out_dir / "train.txt").exists()
    assert (out_dir / "valid.txt").exists()
    assert (out_dir / "test.txt").exists()
    assert (out_dir / "item_count.txt").exists()


def test_preprocess_leave_one_out_split(sample_csv, tmp_path):
    out_dir = tmp_path / "processed"
    preprocess(input_path=sample_csv, output_dir=out_dir, fmt="csv", min_interactions=5)

    train_data, valid_data, test_data, item_count = load_processed_data(out_dir)

    # Each user should appear in all splits
    assert set(train_data.keys()) == set(valid_data.keys()) == set(test_data.keys())

    for uid in train_data:
        full_seq = train_data[uid]
        val_item = valid_data[uid]
        test_item = test_data[uid]
        # val item and test item must be single items (integers)
        assert isinstance(val_item, int)
        assert isinstance(test_item, int)
        # Items must not overlap (last=test, second-to-last=val, rest=train)
        assert test_item not in full_seq
        assert val_item not in full_seq


def test_min_interactions_filter(tmp_path):
    """Users with fewer than min_interactions should be filtered out."""
    csv_file = tmp_path / "sparse.csv"
    # user 1: 6 interactions (kept); user 2: 3 interactions (filtered)
    rows = [f"1,{i},{i}" for i in range(1, 7)]
    rows += [f"2,{i},{i}" for i in range(1, 4)]
    csv_file.write_text("user_id,item_id,timestamp\n" + "\n".join(rows))

    out_dir = tmp_path / "processed"
    preprocess(input_path=csv_file, output_dir=out_dir, fmt="csv", min_interactions=5)

    train_data, _, _, _ = load_processed_data(out_dir)
    assert len(train_data) == 1  # only user 1


def test_item_ids_remapped_to_contiguous(sample_csv, tmp_path):
    out_dir = tmp_path / "processed"
    preprocess(input_path=sample_csv, output_dir=out_dir, fmt="csv", min_interactions=5)
    train_data, valid_data, test_data, item_count = load_processed_data(out_dir)
    # item_count must equal max item id across ALL splits (train + val + test), 1-indexed contiguous
    all_items = (
        {item for seq in train_data.values() for item in seq}
        | set(valid_data.values())
        | set(test_data.values())
    )
    assert max(all_items) == item_count  # 1-indexed contiguous


def test_missing_raw_file_raises(tmp_path):
    out_dir = tmp_path / "processed"
    with pytest.raises(FileNotFoundError):
        preprocess(
            input_path=tmp_path / "nonexistent.csv",
            output_dir=out_dir,
            fmt="csv",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_preprocessor.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement sasrec/data/preprocessor.py**

```python
# sasrec/data/preprocessor.py
"""Preprocess raw interaction files into the standard SASRec split format.

Output format (train.txt, valid.txt, test.txt):
    Each line: user_id item_1 item_2 ... item_n  (space-separated, time-sorted)
    valid.txt and test.txt: each line contains user_id and exactly 1 item.
item_count.txt: single integer — total number of unique items after remapping.
"""
import csv
import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess(
    input_path: Path | str,
    output_dir: Path | str,
    fmt: str = "csv",
    min_interactions: int = 5,
) -> None:
    """Parse raw data, filter, remap IDs, and write train/valid/test splits.

    Args:
        input_path: Path to the raw file.
        fmt: One of "csv", "amazon_csv", "steam_json", "movielens".
        min_interactions: Users and items with fewer interactions are dropped.
        output_dir: Directory to write output files.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If fmt is not supported.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_path}")

    # Load (user_id_raw, item_id_raw, timestamp) triples
    triples = _load_triples(input_path, fmt)

    # Count interactions per user and per item
    user_counts: dict[str, int] = defaultdict(int)
    item_counts: dict[str, int] = defaultdict(int)
    for uid, iid, _ in triples:
        user_counts[uid] += 1
        item_counts[iid] += 1

    valid_users = {u for u, c in user_counts.items() if c >= min_interactions}
    valid_items = {i for i, c in item_counts.items() if c >= min_interactions}

    if len(valid_users) < 10:
        warnings.warn(
            f"Only {len(valid_users)} users remain after filtering with "
            f"min_interactions={min_interactions}. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Build per-user sequences (filtered, sorted by timestamp)
    user_sequences: dict[str, list[str]] = defaultdict(list)
    for uid, iid, ts in sorted(triples, key=lambda x: x[2]):
        if uid in valid_users and iid in valid_items:
            user_sequences[uid].append(iid)

    # Remap item IDs to contiguous integers starting from 1
    all_items_ordered = sorted(
        {iid for seq in user_sequences.values() for iid in seq}
    )
    item_to_int: dict[str, int] = {iid: idx + 1 for idx, iid in enumerate(all_items_ordered)}

    # Remap user IDs to contiguous integers starting from 1
    all_users_ordered = sorted(user_sequences.keys())
    user_to_int: dict[str, int] = {uid: idx + 1 for idx, uid in enumerate(all_users_ordered)}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write splits
    with (
        open(output_dir / "train.txt", "w") as f_train,
        open(output_dir / "valid.txt", "w") as f_valid,
        open(output_dir / "test.txt", "w") as f_test,
    ):
        for uid_raw in all_users_ordered:
            uid_int = user_to_int[uid_raw]
            seq_int = [item_to_int[iid] for iid in user_sequences[uid_raw]]
            # test = last item, valid = second-to-last, train = rest
            test_item = seq_int[-1]
            val_item = seq_int[-2]
            train_seq = seq_int[:-2]
            f_train.write(f"{uid_int} " + " ".join(map(str, train_seq)) + "\n")
            f_valid.write(f"{uid_int} {val_item}\n")
            f_test.write(f"{uid_int} {test_item}\n")

    # Write item count
    (output_dir / "item_count.txt").write_text(str(len(item_to_int)))
    logger.info(
        f"Preprocessed {len(all_users_ordered)} users, {len(item_to_int)} items → {output_dir}"
    )


def load_processed_data(
    data_dir: Path | str,
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int], int]:
    """Load preprocessed splits from disk.

    Returns:
        train_data: {user_id: [item, ...]}
        valid_data: {user_id: item}
        test_data:  {user_id: item}
        item_count: total number of unique items
    """
    data_dir = Path(data_dir)
    train_data: dict[int, list[int]] = {}
    valid_data: dict[int, int] = {}
    test_data: dict[int, int] = {}

    for line in (data_dir / "train.txt").read_text().splitlines():
        parts = list(map(int, line.split()))
        train_data[parts[0]] = parts[1:]

    for line in (data_dir / "valid.txt").read_text().splitlines():
        uid, item = map(int, line.split())
        valid_data[uid] = item

    for line in (data_dir / "test.txt").read_text().splitlines():
        uid, item = map(int, line.split())
        test_data[uid] = item

    item_count = int((data_dir / "item_count.txt").read_text().strip())
    return train_data, valid_data, test_data, item_count


def _load_triples(path: Path, fmt: str) -> list[tuple[str, str, float]]:
    """Load (user_id, item_id, timestamp) triples from a raw file."""
    if fmt == "csv":
        return _load_csv(path)
    elif fmt == "amazon_csv":
        return _load_amazon_csv(path)
    elif fmt == "movielens":
        return _load_movielens(path)
    elif fmt == "steam_json":
        return _load_steam_json(path)
    else:
        raise ValueError(f"Unsupported format: '{fmt}'. Choose from: csv, amazon_csv, movielens, steam_json")


def _load_csv(path: Path) -> list[tuple[str, str, float]]:
    """Load generic CSV with required columns: user_id, item_id, timestamp."""
    triples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append((row["user_id"], row["item_id"], float(row["timestamp"])))
    return triples


def _load_amazon_csv(path: Path) -> list[tuple[str, str, float]]:
    """Load Amazon ratings CSV: user_id,item_id,rating,timestamp (no header)."""
    triples = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4:
                triples.append((row[0], row[1], float(row[3])))
    return triples


def _load_movielens(path: Path) -> list[tuple[str, str, float]]:
    """Load MovieLens ratings.dat: UserID::MovieID::Rating::Timestamp."""
    import zipfile
    triples = []
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            with z.open("ml-1m/ratings.dat") as f:
                for line in f:
                    parts = line.decode().strip().split("::")
                    triples.append((parts[0], parts[1], float(parts[3])))
    else:
        for line in path.read_text().splitlines():
            parts = line.strip().split("::")
            triples.append((parts[0], parts[1], float(parts[3])))
    return triples


def _load_steam_json(path: Path) -> list[tuple[str, str, float]]:
    """Load Steam JSON reviews (gzipped or plain)."""
    import gzip
    triples = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
                triples.append((
                    str(record["username"]),
                    str(record["product_id"]),
                    float(record.get("date", i)),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
    return triples
```

- [ ] **Step 4: Create scripts/preprocess.py**

```python
#!/usr/bin/env python3
"""Preprocess a raw dataset into the standard SASRec split format."""
import argparse
import logging
from pathlib import Path
from sasrec.data.data_info import SUPPORTED_DATASETS, get_dataset_info
from sasrec.data.preprocessor import preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw data for SASRec")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=list(SUPPORTED_DATASETS.keys()),
                       help="Built-in dataset name")
    group.add_argument("--input", type=Path,
                       help="Path to custom CSV file (user_id,item_id,timestamp header required)")
    parser.add_argument("--format", dest="fmt", default="csv",
                        choices=["csv", "amazon_csv", "movielens", "steam_json"],
                        help="File format (only for --input; ignored for --dataset)")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory (default: data/processed/<name>)")
    parser.add_argument("--min_interactions", type=int, default=5)
    args = parser.parse_args()

    if args.dataset:
        info = get_dataset_info(args.dataset)
        input_path = Path("data/raw") / info["raw_filename"]
        fmt = info["format"]
        name = args.dataset
    else:
        input_path = args.input
        fmt = args.fmt
        name = args.input.stem

    output_dir = args.output_dir or Path("data/processed") / name
    preprocess(input_path=input_path, output_dir=output_dir, fmt=fmt,
               min_interactions=args.min_interactions)
    print(f"Done. Processed data saved to: {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_preprocessor.py -v
```
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add sasrec/data/preprocessor.py scripts/preprocess.py tests/test_preprocessor.py
git commit -m "feat: preprocessor for CSV, Amazon, MovieLens, Steam formats"
```

---

## Task 5: Dataset and Sampler

**Files:**
- Create: `sasrec/data/sampler.py`
- Create: `sasrec/data/dataset.py`
- Test: `tests/test_sampler.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests for sampler**

```python
# tests/test_sampler.py
import numpy as np
from sasrec.data.sampler import sample_negative, build_eval_negatives


def test_sample_negative_avoids_history():
    history = {1, 2, 3}
    item_count = 10
    for _ in range(50):
        neg = sample_negative(item_count, history, rng=np.random.default_rng(0))
        assert neg not in history
        assert 1 <= neg <= item_count


def test_build_eval_negatives_shape_and_exclusion():
    train_data = {1: [1, 2, 3], 2: [4, 5, 6]}
    valid_data = {1: 4, 2: 7}
    test_data = {1: 5, 2: 8}
    item_count = 20
    num_neg = 10

    val_negs, test_negs = build_eval_negatives(
        train_data, valid_data, test_data, item_count, num_neg, seed=42
    )

    for uid in [1, 2]:
        full_history = set(train_data[uid]) | {valid_data[uid]} | {test_data[uid]}
        assert len(val_negs[uid]) == num_neg
        assert len(test_negs[uid]) == num_neg
        for neg in val_negs[uid]:
            assert neg not in full_history
        for neg in test_negs[uid]:
            assert neg not in full_history


def test_build_eval_negatives_reproducible():
    train_data = {1: [1, 2, 3]}
    valid_data = {1: 4}
    test_data = {1: 5}
    item_count = 100
    a_val, a_test = build_eval_negatives(train_data, valid_data, test_data, item_count, 10, seed=42)
    b_val, b_test = build_eval_negatives(train_data, valid_data, test_data, item_count, 10, seed=42)
    assert a_val[1] == b_val[1]
    assert a_test[1] == b_test[1]


def test_val_and_test_negatives_use_independent_seeds():
    train_data = {1: [1, 2, 3]}
    valid_data = {1: 4}
    test_data = {1: 5}
    item_count = 100
    val_negs, test_negs = build_eval_negatives(
        train_data, valid_data, test_data, item_count, 10, seed=42
    )
    assert val_negs[1] != test_negs[1]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sampler.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement sasrec/data/sampler.py**

```python
# sasrec/data/sampler.py
"""Negative sampling utilities for training and evaluation."""
import numpy as np


def sample_negative(
    item_count: int,
    history: set[int],
    rng: np.random.Generator,
) -> int:
    """Sample one negative item (1-indexed) not in history via rejection sampling."""
    while True:
        neg = int(rng.integers(1, item_count + 1))
        if neg not in history:
            return neg


def build_eval_negatives(
    train_data: dict[int, list[int]],
    valid_data: dict[int, int],
    test_data: dict[int, int],
    item_count: int,
    num_neg: int,
    seed: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Build fixed negative sets for validation and test evaluation.

    val negatives use `seed`, test negatives use `seed + 1`.
    All negatives exclude the user's full interaction history (train + val + test).

    Returns:
        val_negatives:  {user_id: [neg_item, ...]}  length = num_neg per user
        test_negatives: {user_id: [neg_item, ...]}  length = num_neg per user
    """
    val_negs: dict[int, list[int]] = {}
    test_negs: dict[int, list[int]] = {}

    val_rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng(seed + 1)

    for uid in train_data:
        full_history = (
            set(train_data[uid])
            | {valid_data[uid]}
            | {test_data[uid]}
        )
        val_negs[uid] = [
            sample_negative(item_count, full_history, val_rng)
            for _ in range(num_neg)
        ]
        test_negs[uid] = [
            sample_negative(item_count, full_history, test_rng)
            for _ in range(num_neg)
        ]

    return val_negs, test_negs
```

- [ ] **Step 4: Write failing tests for dataset**

```python
# tests/test_dataset.py
import torch
import pytest
from sasrec.data.dataset import SASRecDataset, SASRecEvalDataset


@pytest.fixture
def small_data():
    # 3 users, each with 7 items in train (after removing val+test)
    # user i has items [i*10+1 .. i*10+7] chronologically
    train_data = {
        1: [11, 12, 13, 14, 15],
        2: [21, 22, 23, 24, 25],
        3: [31, 32, 33, 34, 35],
    }
    valid_data = {1: 16, 2: 26, 3: 36}
    test_data  = {1: 17, 2: 27, 3: 37}
    return train_data, valid_data, test_data


def test_train_dataset_length(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    assert len(ds) == 3


def test_train_dataset_shapes(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    seq, pos, neg = ds[0]
    assert seq.shape == (10,)
    assert pos.shape == (10,)
    assert neg.shape == (10,)


def test_train_seq_left_padded(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    seq, pos, _ = ds[0]
    # 5 train items → seq uses items[:-1] = 4 items, pos uses items[1:] = 4 items
    # Both have 4 non-zero entries and 6 padding zeros
    assert (seq == 0).sum() == 6
    assert (seq != 0).sum() == 4
    assert (pos == 0).sum() == 6
    assert (pos != 0).sum() == 4


def test_pos_is_seq_shifted_left(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    seq, pos, _ = ds[0]
    # pos[i] == seq[i+1] for all non-padding consecutive slots
    for i in range(9):
        if seq[i] != 0 and seq[i + 1] != 0:
            assert pos[i] == seq[i + 1]
    # The last non-zero pos slot holds the last training item
    # (which does NOT appear in seq — it's target-only)
    last_pos_item = pos[pos != 0][-1].item()
    last_train_item = list(train_data.values())[0][-1]
    assert last_pos_item == last_train_item


def test_neg_not_in_user_history(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    for idx in range(len(ds)):
        seq, _, neg = ds[idx]
        user_items = set(seq.tolist()) - {0}
        for n in neg.tolist():
            if n != 0:
                assert n not in user_items


def test_eval_dataset_length(small_data):
    train_data, valid_data, test_data = small_data
    val_negs = {1: [2, 3], 2: [4, 5], 3: [6, 7]}
    ds = SASRecEvalDataset(train_data, valid_data, val_negs, item_count=40, maxlen=10)
    assert len(ds) == 3


def test_eval_dataset_candidate_count(small_data):
    train_data, valid_data, _ = small_data
    val_negs = {uid: list(range(1, 11)) for uid in train_data}  # 10 negs
    ds = SASRecEvalDataset(train_data, valid_data, val_negs, item_count=40, maxlen=10)
    seq, candidates, label_idx = ds[0]
    # candidates = 10 negs + 1 positive = 11
    assert candidates.shape == (11,)
    # label_idx points to the positive item in candidates
    assert candidates[label_idx] == list(valid_data.values())[0]
```

- [ ] **Step 5: Run test to verify it fails**

```bash
pytest tests/test_dataset.py -v
```
Expected: `ImportError`

- [ ] **Step 6: Implement sasrec/data/dataset.py**

```python
# sasrec/data/dataset.py
"""PyTorch Dataset classes for SASRec training and evaluation."""
import numpy as np
import torch
from torch.utils.data import Dataset
from sasrec.data.sampler import sample_negative


class SASRecDataset(Dataset):
    """Training dataset. Each sample is one user's full sequence.

    Returns (seq, pos, neg) tensors of shape [maxlen]:
        seq: input sequence, left-padded with 0.
        pos: positive targets = seq shifted left by 1, last slot = final train item.
        neg: one sampled negative per position (0 where seq is padding).
    """

    def __init__(
        self,
        train_data: dict[int, list[int]],
        item_count: int,
        maxlen: int,
        seed: int = 42,
    ) -> None:
        self.users = sorted(train_data.keys())
        self.train_data = train_data
        self.item_count = item_count
        self.maxlen = maxlen
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid = self.users[idx]
        seq_full = self.train_data[uid]
        history = set(seq_full)

        # Match the original SASRec training protocol exactly:
        #   seq = training_items[:-1]  (all but the last training item, as input)
        #   pos = training_items[1:]   (all but the first training item, as targets)
        # This ensures pos[-1] = last training item ≠ 0, so ALL non-padding positions
        # contribute to the BCE loss. The last training item is only a target, never input.
        seq_raw = seq_full[:-1]  # input:  items 1 .. T-3 (0-indexed)
        pos_raw = seq_full[1:]   # target: items 2 .. T-2 (0-indexed), includes last train item

        # Truncate from the right to fit maxlen
        seq_raw = seq_raw[-(self.maxlen):]
        pos_raw = pos_raw[-(self.maxlen):]
        seq_len = len(seq_raw)

        # Left-pad both to maxlen with 0
        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-seq_len:] = seq_raw
        pos = np.zeros(self.maxlen, dtype=np.int64)
        pos[-seq_len:] = pos_raw

        # Build neg: one sampled negative per position where pos is non-zero
        neg = np.zeros(self.maxlen, dtype=np.int64)
        for i in range(self.maxlen):
            if pos[i] != 0:
                neg[i] = sample_negative(self.item_count, history, self.rng)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(pos),
            torch.from_numpy(neg),
        )


class SASRecEvalDataset(Dataset):
    """Evaluation dataset (val or test). Each sample is one user.

    For validation: input sequence = training items only (excludes val and test).
    For test:       input sequence = training items + val item (excludes test only).
                    Pass `prefix_data=valid_data` to include the val item.

    Returns (seq, candidates, label_idx):
        seq:        input sequence tensor [maxlen], left-padded with 0.
        candidates: item ids tensor [num_neg+1], positive item first.
        label_idx:  index of the positive item in candidates (always 0).
    """

    def __init__(
        self,
        train_data: dict[int, list[int]],
        target_data: dict[int, int],
        neg_samples: dict[int, list[int]],
        item_count: int,
        maxlen: int,
        prefix_data: dict[int, int] | None = None,
    ) -> None:
        self.users = sorted(target_data.keys())
        self.train_data = train_data
        self.target_data = target_data
        self.neg_samples = neg_samples
        self.maxlen = maxlen
        self.prefix_data = prefix_data  # if set, appends one extra item (e.g. val item) to input

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid = self.users[idx]
        seq_raw = list(self.train_data[uid])
        if self.prefix_data is not None and uid in self.prefix_data:
            seq_raw = seq_raw + [self.prefix_data[uid]]
        seq_raw = seq_raw[-self.maxlen:]
        seq_len = len(seq_raw)

        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-seq_len:] = seq_raw

        target = self.target_data[uid]
        negs = self.neg_samples[uid]
        candidates = np.array([target] + negs, dtype=np.int64)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(candidates),
            torch.tensor(0, dtype=torch.long),  # positive is always index 0
        )
```

- [ ] **Step 7: Run all tests to verify they pass**

```bash
pytest tests/test_sampler.py tests/test_dataset.py -v
```
Expected: all passed

- [ ] **Step 8: Commit**

```bash
git add sasrec/data/sampler.py sasrec/data/dataset.py tests/test_sampler.py tests/test_dataset.py
git commit -m "feat: dataset and negative sampler"
```

---

## Task 6: Transformer Modules

**Files:**
- Create: `sasrec/model/modules.py`
- Test: `tests/test_modules.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_modules.py
import torch
import pytest
from sasrec.model.modules import MultiHeadAttention, PointwiseFeedForward


def test_multihead_attention_output_shape():
    B, L, D = 4, 10, 32
    mha = MultiHeadAttention(hidden_units=D, num_heads=2, dropout_rate=0.0)
    x = torch.randn(B, L, D)
    token_ids = torch.randint(1, 20, (B, L))  # no padding
    out = mha(x, token_ids)
    assert out.shape == (B, L, D)


def test_multihead_attention_causal_mask():
    """Output at position t must not depend on positions t+1, ..., L-1."""
    B, L, D = 1, 6, 16
    mha = MultiHeadAttention(hidden_units=D, num_heads=1, dropout_rate=0.0)
    mha.eval()
    x = torch.randn(B, L, D)
    token_ids = torch.ones(B, L, dtype=torch.long)
    out1 = mha(x, token_ids)

    # Modify positions 3+ in x — output at position 2 must not change
    x2 = x.clone()
    x2[:, 3:, :] += 10.0
    out2 = mha(x2, token_ids)
    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5)


def test_multihead_attention_padding_mask():
    """Padding positions (token_id=0) are masked as keys."""
    B, L, D = 2, 8, 16
    mha = MultiHeadAttention(hidden_units=D, num_heads=1, dropout_rate=0.0)
    mha.eval()
    x = torch.randn(B, L, D)
    token_ids = torch.ones(B, L, dtype=torch.long)
    token_ids[:, :4] = 0  # first 4 positions are padding

    out = mha(x, token_ids)
    # Output should be finite (no NaN from softmax over all-masked rows)
    assert torch.isfinite(out).all()


def test_feedforward_output_shape():
    B, L, D = 4, 10, 32
    ffn = PointwiseFeedForward(hidden_units=D, dropout_rate=0.0)
    x = torch.randn(B, L, D)
    out = ffn(x)
    assert out.shape == (B, L, D)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_modules.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement sasrec/model/modules.py**

```python
# sasrec/model/modules.py
"""Reusable Transformer primitives for SASRec.

Architecture follows the original SASRec codebase:
- Pre-LN: LayerNorm applied BEFORE each sublayer (not after).
- ReLU activation in the feed-forward network.
- Causal (lower-triangular) self-attention mask.
- Key padding mask derived from token ids (id == 0 → padding), not from embedding vectors.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causally-masked multi-head self-attention with Pre-LN and key padding mask.

    Padding positions are identified by token_ids == 0 (passed in from the caller),
    which is unambiguous regardless of embedding weight magnitude.
    """

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        assert hidden_units % num_heads == 0, (
            f"hidden_units ({hidden_units}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads

        self.q_proj = nn.Linear(hidden_units, hidden_units, bias=False)
        self.k_proj = nn.Linear(hidden_units, hidden_units, bias=False)
        self.v_proj = nn.Linear(hidden_units, hidden_units, bias=False)
        self.out_proj = nn.Linear(hidden_units, hidden_units, bias=False)

        self.layer_norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [B, L, D] input sequence
            token_ids:  [B, L] original token indices; 0 = padding

        Returns:
            [B, L, D] output after Pre-LN → attention → residual
        """
        residual = x
        x = self.layer_norm(x)

        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        Q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        K = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        scale = math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, L, L]

        # Causal mask: each position can only attend to itself and earlier positions
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Key padding mask: mask out key positions where token_id == 0
        key_padding_mask = (token_ids == 0)  # [B, L], True = padding
        scores = scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        # Replace any all-masked rows with 0 to avoid NaN from softmax
        all_masked = scores.isinf().all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return residual + out


class PointwiseFeedForward(nn.Module):
    """Position-wise feed-forward network with Pre-LN.

    Uses ReLU activation to match the original SASRec codebase.
    """

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_units)
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_modules.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add sasrec/model/modules.py tests/test_modules.py
git commit -m "feat: transformer modules (MultiHeadAttention, PointwiseFeedForward)"
```

---

## Task 7: SASRec Model

**Files:**
- Create: `sasrec/model/sasrec.py`
- Test: `tests/test_sasrec.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sasrec.py
import torch
import pytest
from sasrec.model.sasrec import SASRec


@pytest.fixture
def model():
    return SASRec(item_num=100, hidden_units=32, maxlen=10, num_blocks=2, num_heads=2, dropout_rate=0.0)


def test_forward_shape(model):
    B, L = 4, 10
    seq = torch.randint(0, 101, (B, L))
    logits = model(seq)
    # logits shape: [B, item_num] for prediction at the last non-padding position
    assert logits.shape == (B, 100)


def test_no_nan_in_output(model):
    seq = torch.randint(0, 101, (4, 10))
    logits = model(seq)
    assert torch.isfinite(logits).all()


def test_padding_sequence_output_finite(model):
    """A fully padded sequence should not produce NaN."""
    seq = torch.zeros(2, 10, dtype=torch.long)
    logits = model(seq)
    assert torch.isfinite(logits).all()


def test_score_positive_negative(model):
    """score() method returns dot product with given item embeddings."""
    seq = torch.randint(1, 100, (2, 10))
    pos_items = torch.randint(1, 100, (2, 10))
    neg_items = torch.randint(1, 100, (2, 10))
    pos_logits, neg_logits = model.score(seq, pos_items, neg_items)
    assert pos_logits.shape == (2, 10)
    assert neg_logits.shape == (2, 10)


def test_weight_tying(model):
    """Item embedding table is shared between input lookup and output projection."""
    # The embedding weight used in forward should be the same tensor as in item_emb
    assert model.item_emb.weight is model.item_emb.weight  # trivially same
    # More meaningful: check that changing item_emb changes score output
    seq = torch.randint(1, 50, (1, 10))
    pos = torch.randint(1, 50, (1, 10))
    neg = torch.randint(51, 100, (1, 10))
    p1, n1 = model.score(seq, pos, neg)
    with torch.no_grad():
        model.item_emb.weight[1] += 100.0
    p2, n2 = model.score(seq, pos, neg)
    assert not torch.allclose(p1, p2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sasrec.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement sasrec/model/sasrec.py**

```python
# sasrec/model/sasrec.py
"""SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018).

Architecture:
- Learned item embeddings (padding_idx=0, weight-tied with output)
- Learned 1-indexed positional embeddings (padding_idx=0, maxlen+1 rows)
- N stacked Transformer blocks: Pre-LN → CausalMHA → Residual → Pre-LN → FFN → Residual
- Prediction: last non-padding hidden state dot-producted with item embeddings
"""
import torch
import torch.nn as nn
from sasrec.model.modules import MultiHeadAttention, PointwiseFeedForward


class SASRec(nn.Module):
    def __init__(
        self,
        item_num: int,
        hidden_units: int,
        maxlen: int,
        num_blocks: int,
        num_heads: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.item_num = item_num
        self.maxlen = maxlen

        # Item embedding: index 0 = padding (always zero, no gradient)
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)

        # Positional embedding: 1-indexed (index 0 = padding, suppressed by padding_idx)
        # Row i corresponds to the i-th slot in the fixed-length window (1=oldest, maxlen=newest)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)

        self.emb_dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([
            _TransformerBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_units)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module.padding_idx is not None:
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.weight[module.padding_idx])

    def _encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode an input sequence.

        Args:
            seq: [B, L] token id tensor (0 = padding)

        Returns:
            [B, L, D] hidden states
        """
        B, L = seq.shape
        # Position indices: 1-indexed, same shape as seq
        pos_ids = torch.arange(1, L + 1, device=seq.device).unsqueeze(0).expand(B, -1)
        # Zero out position indices for padding slots
        pos_ids = pos_ids * (seq != 0).long()

        x = self.item_emb(seq) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x, seq)

        x = self.final_norm(x)
        return x

    def _get_last_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """Extract the hidden state at the last non-padding position.

        Returns:
            [B, D]
        """
        x = self._encode(seq)
        # Last non-padding position index for each batch element
        # (seq != 0).sum(dim=1) - 1 gives the 0-indexed position of the last real token
        # But since seq is LEFT-padded, the last real token is always at index L-1
        # if there is at least one non-padding token; otherwise return zeros.
        last_idx = (seq != 0).long().sum(dim=1) - 1  # [B]
        last_idx = last_idx.clamp(min=0)
        # Gather: always take the rightmost real token (index L-1 after left-padding)
        out = x[:, -1, :]  # [B, D] — last position in left-padded sequence = newest item
        return out

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Compute logits over all items for the next-item prediction task.

        Args:
            seq: [B, L] input sequence (left-padded with 0)

        Returns:
            [B, item_num] logit scores where index i → item id (i+1).
            i.e. logits[:, 0] = score for item 1, logits[:, item_num-1] = score for item_num.
        """
        h = self._get_last_hidden(seq)  # [B, D]
        # Weight-tied output: use the item embedding table rows 1..item_num (exclude padding row 0).
        # logits[:, i] corresponds to item id (i+1).
        item_embeddings = self.item_emb.weight[1:]  # [item_num, D], row i = item id (i+1)
        logits = torch.matmul(h, item_embeddings.T)  # [B, item_num]
        return logits

    def score(
        self,
        seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-position scores for training (BCE loss).

        Args:
            seq:       [B, L] input sequence
            pos_items: [B, L] positive target items per position
            neg_items: [B, L] negative sample items per position

        Returns:
            pos_logits: [B, L] dot product scores for positive items
            neg_logits: [B, L] dot product scores for negative items
        """
        h = self._encode(seq)  # [B, L, D]
        pos_emb = self.item_emb(pos_items)  # [B, L, D]
        neg_emb = self.item_emb(neg_items)  # [B, L, D]
        pos_logits = (h * pos_emb).sum(dim=-1)  # [B, L]
        neg_logits = (h * neg_emb).sum(dim=-1)  # [B, L]
        return pos_logits, neg_logits


class _TransformerBlock(nn.Module):
    """Single SASRec Transformer block: Pre-LN MHA + Pre-LN FFN."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(hidden_units, num_heads, dropout_rate)
        self.ffn = PointwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, token_ids)
        x = self.ffn(x)
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sasrec.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add sasrec/model/sasrec.py tests/test_sasrec.py
git commit -m "feat: SASRec model with Pre-LN transformer blocks and weight-tied embeddings"
```

---

## Task 8: Evaluation Metrics

**Files:**
- Create: `sasrec/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import torch
import pytest
from sasrec.evaluation.metrics import ndcg_at_k, hit_at_k, evaluate_batch


def test_hit_at_k_positive_ranked_first():
    # scores: positive=10.0, negatives all < 10
    scores = torch.tensor([[10.0, 1.0, 2.0, 3.0]])  # [1, 4]
    label_idx = torch.tensor([0])
    assert hit_at_k(scores, label_idx, k=3) == 1.0


def test_hit_at_k_positive_not_in_top_k():
    scores = torch.tensor([[1.0, 5.0, 6.0, 7.0]])
    label_idx = torch.tensor([0])
    assert hit_at_k(scores, label_idx, k=2) == 0.0


def test_ndcg_at_k_positive_ranked_first():
    scores = torch.tensor([[10.0, 1.0, 2.0, 3.0]])
    label_idx = torch.tensor([0])
    # Positive at rank 1: NDCG = 1/log2(2) = 1.0
    result = ndcg_at_k(scores, label_idx, k=10)
    assert abs(result - 1.0) < 1e-5


def test_ndcg_at_k_positive_ranked_second():
    scores = torch.tensor([[5.0, 10.0, 1.0, 2.0]])
    label_idx = torch.tensor([0])
    # Positive at rank 2: NDCG = 1/log2(3)
    import math
    result = ndcg_at_k(scores, label_idx, k=10)
    assert abs(result - 1.0 / math.log2(3)) < 1e-5


def test_ndcg_at_k_positive_outside_top_k():
    scores = torch.tensor([[1.0, 10.0, 9.0, 8.0, 7.0]])
    label_idx = torch.tensor([0])
    assert ndcg_at_k(scores, label_idx, k=2) == 0.0


def test_evaluate_batch_returns_dict():
    scores = torch.randn(8, 101)
    label_idx = torch.zeros(8, dtype=torch.long)
    results = evaluate_batch(scores, label_idx, k_values=[5, 10, 20])
    assert "NDCG@5" in results
    assert "HR@5" in results
    assert "NDCG@10" in results
    assert "HR@10" in results
    assert "NDCG@20" in results
    assert "HR@20" in results


def test_evaluate_batch_values_in_range():
    scores = torch.randn(16, 101)
    label_idx = torch.zeros(16, dtype=torch.long)
    results = evaluate_batch(scores, label_idx, k_values=[10])
    assert 0.0 <= results["NDCG@10"] <= 1.0
    assert 0.0 <= results["HR@10"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement sasrec/evaluation/metrics.py**

```python
# sasrec/evaluation/metrics.py
"""NDCG@K and HR@K metrics for sequential recommendation evaluation.

Evaluation modes:
- sampled: candidate set = {positive} + {num_neg_eval negatives}. Fast.
- full:    candidate set = {positive} + {all items not in user's history except positive}.

In both modes the positive item is always at index 0 in the candidates tensor.
"""
import math
import torch


def hit_at_k(scores: torch.Tensor, label_idx: torch.Tensor, k: int) -> float:
    """Hit Rate @ K averaged over a batch.

    Args:
        scores:    [B, num_candidates] — higher = more relevant
        label_idx: [B] — index of the positive item in each row of scores
        k:         cutoff

    Returns:
        Mean HR@K over the batch.
    """
    B = scores.shape[0]
    # Get the rank of the positive item (0-indexed, lower = better)
    _, top_k_indices = scores.topk(k, dim=1)  # [B, k]
    label_idx_expanded = label_idx.unsqueeze(1).expand_as(top_k_indices)  # [B, k]
    hits = (top_k_indices == label_idx_expanded).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(scores: torch.Tensor, label_idx: torch.Tensor, k: int) -> float:
    """NDCG @ K averaged over a batch.

    Args:
        scores:    [B, num_candidates]
        label_idx: [B]
        k:         cutoff

    Returns:
        Mean NDCG@K over the batch.
    """
    B = scores.shape[0]
    # Rank of each item (0-indexed, lower = better)
    # argsort descending gives the item at each rank position
    sorted_indices = scores.argsort(dim=1, descending=True)  # [B, num_candidates]

    ndcg_values = []
    for b in range(B):
        pos_idx = label_idx[b].item()
        rank_positions = (sorted_indices[b] == pos_idx).nonzero(as_tuple=True)[0]
        rank = rank_positions[0].item()  # 0-indexed rank
        if rank < k:
            ndcg_values.append(1.0 / math.log2(rank + 2))  # rank+2 because log2(rank+1) with 1-indexed
        else:
            ndcg_values.append(0.0)

    return sum(ndcg_values) / B


def evaluate_batch(
    scores: torch.Tensor,
    label_idx: torch.Tensor,
    k_values: list[int],
) -> dict[str, float]:
    """Compute NDCG@K and HR@K for multiple K values.

    Args:
        scores:    [B, num_candidates]
        label_idx: [B]
        k_values:  list of K values to evaluate

    Returns:
        Dict like {"NDCG@10": 0.42, "HR@10": 0.65, ...}
    """
    results: dict[str, float] = {}
    for k in k_values:
        results[f"NDCG@{k}"] = ndcg_at_k(scores, label_idx, k)
        results[f"HR@{k}"] = hit_at_k(scores, label_idx, k)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add sasrec/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: NDCG@K and HR@K evaluation metrics"
```

---

## Task 9: Trainer

**Files:**
- Create: `sasrec/trainer/trainer.py`
- Create: `sasrec/utils.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Create sasrec/utils.py (seed helper)**

```python
# sasrec/utils.py
"""Shared utilities."""
import random
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets Python, NumPy, PyTorch CPU and GPU seeds.
    Enables deterministic cuDNN behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        logger.warning("torch.use_deterministic_algorithms(True) not available; skipping.")


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
```

- [ ] **Step 2: Write failing integration test**

```python
# tests/test_trainer.py
import torch
import pytest
from pathlib import Path
from sasrec.model.sasrec import SASRec
from sasrec.trainer.trainer import Trainer
from sasrec.data.sampler import build_eval_negatives
from sasrec.data.dataset import SASRecDataset, SASRecEvalDataset


@pytest.fixture
def tiny_setup(tmp_path):
    item_count = 20
    train_data = {i: list(range(1, 8)) for i in range(1, 6)}  # 5 users, 7 train items each
    valid_data = {i: 8 for i in range(1, 6)}
    test_data  = {i: 9 for i in range(1, 6)}

    val_negs, test_negs = build_eval_negatives(
        train_data, valid_data, test_data, item_count, num_neg=5, seed=42
    )

    train_ds = SASRecDataset(train_data, item_count=item_count, maxlen=10)
    val_ds   = SASRecEvalDataset(train_data, valid_data, val_negs, item_count, maxlen=10)
    # Test input includes val item as context (leave-one-out: exclude only test item)
    test_ds  = SASRecEvalDataset(train_data, test_data, test_negs, item_count, maxlen=10, prefix_data=valid_data)

    model = SASRec(item_num=item_count, hidden_units=16, maxlen=10, num_blocks=1, num_heads=1, dropout_rate=0.0)

    return model, train_ds, val_ds, test_ds, tmp_path


def test_trainer_runs_one_epoch(tiny_setup):
    model, train_ds, val_ds, test_ds, tmp_path = tiny_setup
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        checkpoint_dir=tmp_path,
        k_values=[5],
        lr=0.001,
        batch_size=4,
        num_epochs=1,
        patience=5,
        use_wandb=False,
        log_dir=str(tmp_path / "runs"),
        seed=42,
    )
    trainer.train()


def test_trainer_saves_checkpoints(tiny_setup):
    model, train_ds, val_ds, test_ds, tmp_path = tiny_setup
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        checkpoint_dir=tmp_path,
        k_values=[5],
        lr=0.001,
        batch_size=4,
        num_epochs=2,
        patience=5,
        use_wandb=False,
        log_dir=str(tmp_path / "runs"),
        seed=42,
    )
    trainer.train()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "latest.pt").exists()


def test_trainer_evaluate_returns_metrics(tiny_setup):
    model, train_ds, val_ds, test_ds, tmp_path = tiny_setup
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        checkpoint_dir=tmp_path,
        k_values=[5],
        lr=0.001,
        batch_size=4,
        num_epochs=1,
        patience=5,
        use_wandb=False,
        log_dir=str(tmp_path / "runs"),
        seed=42,
    )
    metrics = trainer.evaluate(val_ds)
    assert "NDCG@5" in metrics
    assert "HR@5" in metrics
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_trainer.py -v
```
Expected: `ImportError`

- [ ] **Step 4: Implement sasrec/trainer/trainer.py**

```python
# sasrec/trainer/trainer.py
"""Training loop with checkpoint saving, early stopping, TensorBoard and optional wandb."""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sasrec.data.dataset import SASRecDataset, SASRecEvalDataset
from sasrec.evaluation.metrics import evaluate_batch
from sasrec.model.sasrec import SASRec
from sasrec.utils import set_seed, worker_init_fn

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: SASRec,
        train_dataset: SASRecDataset,
        val_dataset: SASRecEvalDataset,
        test_dataset: SASRecEvalDataset,
        checkpoint_dir: str | Path,
        k_values: list[int],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.98,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        num_epochs: int = 200,
        patience: int = 20,
        use_wandb: bool = False,
        wandb_project: str = "sasrec",
        log_dir: str = "runs/",
        seed: int = 42,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.k_values = k_values
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.use_wandb = use_wandb

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.writer = SummaryWriter(log_dir=log_dir)

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError as e:
                raise ImportError(
                    "wandb is not installed. Install with: pip install -e '.[logging]'"
                ) from e
            self._wandb.init(project=wandb_project)
        else:
            self._wandb = None

    def train(self) -> None:
        set_seed(self.seed)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=worker_init_fn,
        )

        best_ndcg = -1.0
        epochs_without_improvement = 0
        # Prefer NDCG@10 as the primary selection metric (matching the paper);
        # fall back to the first k value if 10 is not in the list.
        primary_k = 10 if 10 in self.k_values else self.k_values[0]

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(self.val_dataset)

            logger.info(
                f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )

            # TensorBoard
            self.writer.add_scalar("train/loss", train_loss, epoch)
            for name, val in val_metrics.items():
                self.writer.add_scalar(f"val/{name}", val, epoch)

            # wandb
            if self._wandb:
                self._wandb.log({"epoch": epoch, "train_loss": train_loss, **{f"val/{k}": v for k, v in val_metrics.items()}})

            # Checkpointing
            torch.save(self.model.state_dict(), self.checkpoint_dir / "latest.pt")
            current_ndcg = val_metrics.get(f"NDCG@{primary_k}", 0.0)
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break

        # Final test evaluation on best model
        self.model.load_state_dict(torch.load(self.checkpoint_dir / "best.pt", map_location=self.device))
        test_metrics = self.evaluate(self.test_dataset)
        logger.info("Test metrics: " + " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
        if self._wandb:
            self._wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        self.writer.close()

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        try:
            for seq, pos, neg in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
                seq = seq.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)

                pos_logits, neg_logits = self.model.score(seq, pos, neg)
                istarget = (pos != 0).float()

                pos_loss = self.criterion(pos_logits, torch.ones_like(pos_logits))
                neg_loss = self.criterion(neg_logits, torch.zeros_like(neg_logits))
                loss = ((pos_loss + neg_loss) * istarget).sum() / istarget.sum()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        except Exception:
            logging.exception("Error during training epoch")
            raise
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, dataset: SASRecEvalDataset) -> dict[str, float]:
        """Evaluate the model on a dataset, returning NDCG@K and HR@K metrics."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        all_scores = []
        all_label_idx = []

        for seq, candidates, label_idx in loader:
            seq = seq.to(self.device)
            candidates = candidates.to(self.device)

            # Score each candidate item using the model's last hidden state
            h = self.model._get_last_hidden(seq)  # [B, D]
            cand_emb = self.model.item_emb(candidates)  # [B, num_cands, D]
            scores = torch.bmm(cand_emb, h.unsqueeze(-1)).squeeze(-1)  # [B, num_cands]

            all_scores.append(scores.cpu())
            all_label_idx.append(label_idx.cpu())

        all_scores = torch.cat(all_scores, dim=0)
        all_label_idx = torch.cat(all_label_idx, dim=0)

        return evaluate_batch(all_scores, all_label_idx, self.k_values)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_trainer.py -v
```
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add sasrec/utils.py sasrec/trainer/trainer.py tests/test_trainer.py
git commit -m "feat: trainer with BCE loss, early stopping, TensorBoard and checkpoint saving"
```

---

## Task 10: Main Experiment Script

**Files:**
- Create: `scripts/run_experiment.py`

- [ ] **Step 1: Implement scripts/run_experiment.py**

```python
#!/usr/bin/env python3
"""Main entry point for training and evaluating SASRec."""
import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from sasrec.config import load_config
from sasrec.data.preprocessor import load_processed_data
from sasrec.data.sampler import build_eval_negatives
from sasrec.data.dataset import SASRecDataset, SASRecEvalDataset
from sasrec.model.sasrec import SASRec
from sasrec.trainer.trainer import Trainer
from sasrec.data.data_info import SUPPORTED_DATASETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SASRec")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dataset", choices=list(SUPPORTED_DATASETS.keys()))
    group.add_argument("--data_dir", type=Path, help="Path to preprocessed data directory")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf overrides, e.g. model.hidden_units=64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(dataset=args.dataset, overrides=args.overrides or None)
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    # Resolve data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif args.dataset:
        data_dir = Path("data/processed") / args.dataset
    else:
        logger.error("Provide --dataset or --data_dir")
        sys.exit(1)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}. Run preprocess.py first.")
        sys.exit(1)

    # Load data
    train_data, valid_data, test_data, item_count = load_processed_data(data_dir)
    logger.info(f"Loaded {len(train_data)} users, {item_count} items from {data_dir}")

    # Build fixed evaluation negatives
    val_negs, test_negs = build_eval_negatives(
        train_data, valid_data, test_data,
        item_count=item_count,
        num_neg=cfg.data.num_neg_eval,
        seed=cfg.train.seed,
    )

    # Datasets
    train_ds = SASRecDataset(train_data, item_count=item_count,
                              maxlen=cfg.data.maxlen, seed=cfg.train.seed)
    val_ds   = SASRecEvalDataset(train_data, valid_data, val_negs,
                                  item_count=item_count, maxlen=cfg.data.maxlen)
    # Test input includes the val item as context (leave-one-out: only test item excluded)
    test_ds  = SASRecEvalDataset(train_data, test_data, test_negs,
                                  item_count=item_count, maxlen=cfg.data.maxlen,
                                  prefix_data=valid_data)

    # Model
    model = SASRec(
        item_num=item_count,
        hidden_units=cfg.model.hidden_units,
        maxlen=cfg.data.maxlen,
        num_blocks=cfg.model.num_blocks,
        num_heads=cfg.model.num_heads,
        dropout_rate=cfg.model.dropout_rate,
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")

    # Trainer
    checkpoint_dir = Path(cfg.logging.checkpoint_dir) / (args.dataset or data_dir.name)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        checkpoint_dir=checkpoint_dir,
        k_values=list(cfg.eval.k_values),
        lr=cfg.train.lr,
        beta1=cfg.train.beta1,
        beta2=cfg.train.beta2,
        weight_decay=cfg.train.weight_decay,
        batch_size=cfg.train.batch_size,
        num_epochs=cfg.train.num_epochs,
        patience=cfg.train.patience,
        use_wandb=cfg.logging.use_wandb,
        wandb_project=cfg.logging.wandb_project,
        log_dir=cfg.logging.log_dir,
        seed=cfg.train.seed,
    )

    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script (dry run with tiny synthetic data)**

```bash
# Create tiny synthetic processed data for smoke test
mkdir -p data/processed/smoke_test
python -c "
# 5 users, 10 items each (train=8 items, val=1, test=1)
import pathlib, random
d = pathlib.Path('data/processed/smoke_test')
d.mkdir(exist_ok=True)
random.seed(0)
items = list(range(1, 11))
(d/'train.txt').write_text('\n'.join(f'{u} ' + ' '.join(map(str,items[:8])) for u in range(1,6)))
(d/'valid.txt').write_text('\n'.join(f'{u} 9' for u in range(1,6)))
(d/'test.txt').write_text('\n'.join(f'{u} 10' for u in range(1,6)))
(d/'item_count.txt').write_text('10')
print('done')
"
python scripts/run_experiment.py --data_dir data/processed/smoke_test train.num_epochs=2 train.batch_size=2
# Note: OmegaConf overrides are passed as positional args (key=value, no -- prefix)
```

Expected: trains for 2 epochs, saves `checkpoints/smoke_test/best.pt`, prints test metrics.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat: run_experiment.py main training script"
```

---

## Task 11: Full Test Suite and README

**Files:**
- Modify: `tests/` (run all tests)
- Create: `README.md`

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests pass. If any fail, fix the failures before proceeding.

- [ ] **Step 2: Create README.md**

```markdown
# SASRec — Python 3.10 / PyTorch Implementation

Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018).
Faithful reproduction of the original paper in Python 3.10 with PyTorch.

## Quick Start

### 1. Install

```bash
pip install -e .
# With wandb support:
pip install -e ".[logging]"
```

### 2. Get data

```bash
python scripts/show_data_info.py --dataset beauty
# Follow the printed instructions to download and place in data/raw/
```

### 3. Preprocess

```bash
python scripts/preprocess.py --dataset beauty
```

### 4. Train

```bash
python scripts/run_experiment.py --dataset beauty
```

Results are logged to `runs/` (TensorBoard) and checkpoints saved to `checkpoints/beauty/`.

View training curves:
```bash
tensorboard --logdir runs/
```

### 5. Override hyperparameters

Pass OmegaConf dot-notation overrides as positional arguments after the named flags:

```bash
python scripts/run_experiment.py --dataset beauty model.hidden_units=64 train.lr=0.0005
```

### 6. Custom dataset

Your CSV must have a header row with columns `user_id`, `item_id`, `timestamp`.

```bash
python scripts/preprocess.py --input my_data.csv --format csv
python scripts/run_experiment.py --data_dir data/processed/my_data
```

### 7. Enable wandb

```bash
python scripts/run_experiment.py --dataset beauty logging.use_wandb=true
```

## Supported Datasets

| Name | Source |
|---|---|
| `beauty` | Amazon Beauty |
| `video_games` | Amazon Video Games |
| `steam` | Steam Reviews |
| `ml-1m` | MovieLens 1M |

## Project Structure

```
sasrec/
├── configs/         # YAML hyperparameters
├── sasrec/          # Python package
│   ├── data/        # Preprocessing, Dataset, Sampler
│   ├── model/       # SASRec model + Transformer modules
│   ├── trainer/     # Training loop
│   └── evaluation/  # Metrics (NDCG@K, HR@K)
└── scripts/         # CLI entry points
```

## Differences from Original

| Original (Python 2 / TF 1.x) | This rewrite |
|---|---|
| Python 2 only | Python 3.10 |
| TensorFlow 1.x | PyTorch ≥ 2.0 |
| No checkpoint saving | `best.pt` and `latest.pt` |
| Bare `except` hides errors | `logging.exception()` + re-raise |
| Hardcoded 101 eval candidates | Configurable via `num_neg_eval` |
| No config file | OmegaConf YAML + CLI overrides |
| No TensorBoard / wandb | Both supported |
| Non-reproducible evaluation | Fixed seeds for eval negatives |
```

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: README with quick start and usage guide"
```

---

## Done ✓

After all tasks complete:

1. Run `pytest tests/ -v` — all green
2. Run the smoke test: `python scripts/run_experiment.py --data_dir data/processed/smoke_test train.num_epochs=2`
3. View TensorBoard: `tensorboard --logdir runs/`
