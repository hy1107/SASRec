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

            self.writer.add_scalar("train/loss", train_loss, epoch)
            for name, val in val_metrics.items():
                self.writer.add_scalar(f"val/{name}", val, epoch)

            if self._wandb:
                self._wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                })

            # Save latest checkpoint every epoch
            torch.save(self.model.state_dict(), self.checkpoint_dir / "latest.pt")

            # Save best checkpoint by NDCG@primary_k
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
        self.model.load_state_dict(
            torch.load(self.checkpoint_dir / "best.pt", map_location=self.device, weights_only=True)
        )
        test_metrics = self.evaluate(self.test_dataset)
        logger.info(
            "Test metrics: " + " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
        )
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
                loss = ((pos_loss + neg_loss) * istarget).sum() / istarget.sum().clamp(min=1)

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

            h = self.model._get_last_hidden(seq)  # [B, D]
            cand_emb = self.model.item_emb(candidates)  # [B, num_cands, D]
            scores = torch.bmm(cand_emb, h.unsqueeze(-1)).squeeze(-1)  # [B, num_cands]

            all_scores.append(scores.cpu())
            all_label_idx.append(label_idx.cpu())

        all_scores = torch.cat(all_scores, dim=0)
        all_label_idx = torch.cat(all_label_idx, dim=0)

        return evaluate_batch(all_scores, all_label_idx, self.k_values)
