import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_classes=8, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ParcelModel(nn.Module):
    def __init__(self, encoder, transformer, classifier):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.classifier = classifier

    def forward(self, x, day_of_year):
        x = self.encoder(x)  # (B, T, F)
        x = self.transformer(x, day_of_year)  # (B, F)
        return self.classifier(x)