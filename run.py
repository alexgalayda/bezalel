import hydra
from omegaconf import DictConfig

from bezalel.dataset import ProteinDataset, ProteinDatasetConfig
from bezalel.models import get_model_from_config
from bezalel.test import test
from bezalel.train import train
from bezalel.utils import get_loss, get_metrics, get_optimizer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    model = get_model_from_config(cfg)
    criterion = get_loss(cfg)
    metrics = get_metrics(cfg)
    match cfg.mode:
        case "train":
            optimizer = get_optimizer(model, cfg)
            train_dataset_config = ProteinDatasetConfig(
                init_qdrant=True,
                processed_dir=cfg.train.train_path,
                **cfg.dataset)
            val_dataset_config = ProteinDatasetConfig(
                init_qdrant=False,
                processed_dir=cfg.train.val_path,
                **cfg.dataset)
            train_dataset = ProteinDataset(train_dataset_config)
            val_dataset = ProteinDataset(val_dataset_config)
            val_dataset.qdrant = train_dataset.qdrant
            train(
                model,
                train_dataset,
                val_dataset,
                criterion,
                optimizer,
                metrics,
                cfg.train,
            )
        case "test":
            test_dataset_config = ProteinDatasetConfig(
                init_qdrant=True,
                processed_dir=cfg.test.test_path,
                **cfg.dataset)
            test_dataset = ProteinDataset(test_dataset_config)
            test(model, test_dataset, criterion, metrics, cfg.test)
        case _:
            raise ValueError(f"Invalid mode: {cfg.mode}")


if __name__ == "__main__":
    main()
