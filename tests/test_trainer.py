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
    test_ds  = SASRecEvalDataset(
        train_data, test_data, test_negs, item_count, maxlen=10, prefix_data=valid_data
    )

    model = SASRec(
        item_num=item_count, hidden_units=16, maxlen=10,
        num_blocks=1, num_heads=1, dropout_rate=0.0
    )

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
