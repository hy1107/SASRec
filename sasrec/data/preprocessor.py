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

    triples = _load_triples(input_path, fmt)

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
        raise ValueError(
            f"Unsupported format: '{fmt}'. Choose from: csv, amazon_csv, movielens, steam_json"
        )


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
