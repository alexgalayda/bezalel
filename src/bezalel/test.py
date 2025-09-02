from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from bezalel.checkpoint_saver import (
    get_last_checkpoint,
    load_model_checkpoint,
)


class TestConfig(BaseModel):
    device: str = "cuda"
    log_dir: Path = Path("logs")
    checkpoint_path: Path = Path("checkpoints")


def test_epoch(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    metrics: dict[str, nn.Module],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for idx, (feat, target) in enumerate(test_loader):
            feat = feat.to(device)
            target = target.to(device)
            outputs = model(feat[0])
            outputs = outputs.squeeze(-1).unsqueeze(0)
            loss = criterion(outputs, target.float())
            total_loss += loss.item()
            if idx % 10 == 0:
                logger.info(
                    f"Test Batch {idx}/{len(test_loader)}, Loss: {loss.item():.4f}"
                )
            for _, metric in metrics.items():
                metric.update(outputs, target)
    metrics = {
        metric_name: metric.compute()
        for metric_name, metric in metrics.items()
    }
    for metric_name, metric in metrics.items():
        logger.info(f"Test {metric_name}: {metric:.4f}")
    return {
        "loss": total_loss / len(test_loader),
        **metrics,
    }


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    metrics: dict[str, nn.Module],
    config: DictConfig,
):
    config = TestConfig(**config)
    if config.checkpoint_path.suffix != ".pth":
        config.checkpoint_path = get_last_checkpoint(config.checkpoint_path)
    model, _, _, _ = load_model_checkpoint(
        config.checkpoint_path, model, map_location=config.device
    )
    config.log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    logger.info("Starting test evaluation...")
    test_metrics = test_epoch(
        model, test_loader, criterion, metrics, torch.device(config.device)
    )
    for metric_name, metric_value in test_metrics.items():
        writer.add_scalar(f"Test/{metric_name}", metric_value, 0)
        logger.info(f"Test {metric_name}: {metric_value:.4f}")
    writer.close()
    logger.info("Testing completed!")
    logger.info(f"Test loss: {test_metrics['loss']:.4f}")
