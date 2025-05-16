import torch

from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix
)

class ClassificationMetrics:
    def __init__(self, num_classes, device="cpu"):
        self.device = device
        self.accuracy = Accuracy(task="multiclass", average='micro', num_classes=num_classes).to(device)
        self.f1_micro = F1Score(task="multiclass", average='micro', num_classes=num_classes).to(device)
        self.f1_weighted = F1Score(task="multiclass", average='weighted', num_classes=num_classes).to(device)
        self.precision = Precision(task="multiclass", average='weighted', num_classes=num_classes).to(device)
        self.recall = Recall(task="multiclass", average='weighted', num_classes=num_classes).to(device)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    def update(self, preds, targets):
        self.accuracy.update(preds, targets)
        self.f1_micro.update(preds, targets)
        self.f1_weighted.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.confmat.update(preds, targets)

    def compute(self):
        return {
            "accuracy": self.accuracy.compute().item(),
            "f1_micro": self.f1_micro.compute().item(),
            "f1_weighted": self.f1_weighted.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "confusion_matrix": self.confmat.compute()
        }

    def reset(self):
        self.accuracy.reset()
        self.f1_micro.reset()
        self.f1_weighted.reset()
        self.precision.reset()
        self.recall.reset()
        self.confmat.reset()
