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
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf overrides as positional args, e.g. model.hidden_units=64 train.lr=0.0005"
    )
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
    train_ds = SASRecDataset(
        train_data, item_count=item_count,
        maxlen=cfg.data.maxlen, seed=cfg.train.seed
    )
    val_ds = SASRecEvalDataset(
        train_data, valid_data, val_negs,
        item_count=item_count, maxlen=cfg.data.maxlen
    )
    # Test input includes the val item as context (leave-one-out: only test item excluded)
    test_ds = SASRecEvalDataset(
        train_data, test_data, test_negs,
        item_count=item_count, maxlen=cfg.data.maxlen,
        prefix_data=valid_data
    )

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
