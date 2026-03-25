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
