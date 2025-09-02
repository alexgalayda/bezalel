import torch
from torchmetrics import Accuracy


class BinaryAccuracyUpper(Accuracy):
    def _get_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

    def update(self, pred: torch.Tensor, y_true: torch.Tensor):
        mask = self._get_mask(pred.shape[0])
        pred = pred[mask]
        y_true = y_true[mask]
        return super().update(pred, y_true)
