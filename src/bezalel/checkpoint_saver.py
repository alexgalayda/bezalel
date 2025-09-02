from pathlib import Path

import torch
from loguru import logger
from pydantic import BaseModel
from torch import nn, optim

from bezalel.utils import timestampt


class CheckpointSaverConfig(BaseModel):
    save_dir: Path = Path("checkpoints")


class CheckpointSaver:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: CheckpointSaverConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = config.save_dir
        self.best_loss = float("inf")
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def save(self, epoch, val_loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            path = self.save_dir / f"{timestampt()}.pth"
            torch.save(checkpoint, path)
            logger.info(
                f"Saved new best checkpoint: '{path}' (epoch {epoch}, loss {val_loss:.4f})"
            )

    def resume(
        self, name: str | None = None, map_location: str | None = None
    ) -> tuple[int, float]:
        if name is None:
            path = get_last_checkpoint(self.save_dir)
        else:
            path = self.save_dir / name
            if not path.exists():
                logger.info("No checkpoint found, starting from scratch.")
                return 0, None
        self.model, self.optimizer, start_epoch, val_loss = (
            load_model_checkpoint(
                path, self.model, self.optimizer, map_location
            )
        )
        logger.info(
            f"Resumed from checkpoint '{path}' at epoch {start_epoch}, best_loss={self.best_loss:.4f}"
        )
        return start_epoch, val_loss


def get_last_checkpoint(save_dir: Path) -> Path:
    checkpoints = list(save_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found")
    return max(checkpoints, key=lambda x: x.stat().st_mtime)


def load_model_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    map_location: str | None = None,
) -> tuple[nn.Module, optim.Optimizer, int, float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", -1) + 1
        val_loss = checkpoint.get("val_loss", float("inf"))
        logger.info(
            f"Loaded model from checkpoint: {checkpoint_path} at epoch {epoch}, val loss {val_loss:.4f}"
        )
    else:
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
    return model, optimizer, epoch, val_loss
