import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import ujson as json
from omegaconf import DictConfig
from torch import nn, optim

from bezalel.metrics import BinaryAccuracyUpper


def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def root_dir() -> Path:
    return Path(__file__).parent.parent.parent.absolute()


def get_optimizer(model: nn.Module, config: DictConfig) -> optim.Optimizer:
    match config.optimizer.name:
        case "Adam":
            optimizer = optim.Adam
        case _:
            raise ValueError(
                f"Invalid optimizer name: {config.optimizer.name}"
            )
    optimizer = optimizer(
        model.parameters(), lr=config.optimizer.learning_rate
    )
    return optimizer


def get_loss(config: DictConfig) -> nn.Module:
    if len(config.loss.names) != len(config.loss.alphas):
        raise ValueError("Loss names and alphas must have the same length")
    losses = []
    for loss_name in config.loss.names:
        match loss_name:
            case "bce":
                losses.append(nn.BCELoss())
            case _:
                raise ValueError(f"Invalid loss name: {loss_name}")

    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss, alpha in zip(losses, config.loss.alphas, strict=False):
            loss_value = loss(pred, target)
            total_loss += alpha * loss_value
        return total_loss

    return loss

def get_metrics(config: DictConfig) -> dict[str, nn.Module]:
    metrics = {}
    for metric_name in config.metrics:
        match metric_name:
            case "accuracy_upper":
                metrics[metric_name] = BinaryAccuracyUpper(**config.accuracy_upper).to(
                    config.device
                )
            case _:
                raise ValueError(f"Invalid metric name: {metric_name}")
    return metrics


def timestampt() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
