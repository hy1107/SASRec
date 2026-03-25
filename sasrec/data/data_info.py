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
