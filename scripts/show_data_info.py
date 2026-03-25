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
