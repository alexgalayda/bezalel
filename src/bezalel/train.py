from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from bezalel.checkpoint_saver import CheckpointSaver, CheckpointSaverConfig


class TrainConfig(BaseModel):
    num_epochs: int
    device: str = "cuda"
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    for idx, (feat, target) in enumerate(train_loader):
        feat = feat.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs = model(feat[0])
        outputs = outputs.squeeze(-1).unsqueeze(0)
        loss = criterion(outputs, target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if idx % 10 == 0:
            logger.info(
                f"Epoch {epoch}, Batch {idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )
    return {
        "loss": total_loss / len(train_loader),
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    metrics: dict[str, nn.Module],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for feat, target in val_loader:
            feat = feat.to(device)
            target = target.to(device)
            outputs = model(feat)
            outputs = model(feat[0])
            outputs = outputs.squeeze(-1).unsqueeze(0)
            loss = criterion(outputs, target.float())
            total_loss += loss.item()
            for _, metric in metrics.items():
                metric.update(outputs, target)
    metrics = {
        metric_name: metric.compute()
        for metric_name, metric in metrics.items()
    }
    for metric_name, metric in metrics.items():
        logger.info(f"Validation {metric_name}: {metric:.4f}")
    return {
        "loss": total_loss / len(val_loader),
        **metrics,
    }


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    metrics: dict[str, nn.Module],
    config: DictConfig,
) -> None:
    config = TrainConfig(**config)
    config.log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    checkpoint_saver = CheckpointSaver(
        model, optimizer, CheckpointSaverConfig(save_dir=config.checkpoint_dir)
    )
    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device,
            epoch,
        )
        val_metrics = validate(
            model, val_loader, criterion, metrics, config.device
        )
        checkpoint_saver.save(epoch, val_metrics["loss"])
        writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/Validation", val_metrics["loss"], epoch)
        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(
                f"Metrics/{metric_name}/Train", metric_value, epoch
            )
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(
                f"Metrics/{metric_name}/Validation", metric_value, epoch
            )
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
    writer.close()
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {val_metrics['loss']:.4f}")
