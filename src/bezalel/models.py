from typing import Literal

import torch
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import nn
from torch.nn import LayerNorm
from torchvision.ops import MLP


class MLPConfig(BaseModel):
    in_channels: int
    hidden_channels: list[int]
    out_channels: int = 1


class MLPSigmoidConfig(BaseModel):
    in_channels: int
    hidden_channels: list[int]
    out_channels: int = 1


class MLPSigmoid(MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(x))


def get_model(
    model_name: Literal["mlp"], config: DictConfig
) -> nn.Module:
    match model_name:
        case "mlp":
            config = MLPConfig(**config).model_dump()
            return MLP(
                in_channels=config["in_channels"],
                hidden_channels=config["hidden_channels"] + [config["out_channels"]],
                norm_layer=LayerNorm
            )
        case "mlp_sigmoid":
            config = MLPSigmoidConfig(**config).model_dump()
            return MLPSigmoid(
                in_channels=config["in_channels"],
                hidden_channels=config["hidden_channels"] + [config["out_channels"]],
                norm_layer=LayerNorm
            )
        case _:
            raise ValueError(f"Invalid model name: {model_name}")


def get_model_from_config(config: DictConfig) -> nn.Module:
    model_name = config.model.name
    if model_config := config.get(model_name):
        return get_model(model_name, model_config).to(config.model.device)
    raise ValueError(f"Model config not found for {model_name}")
