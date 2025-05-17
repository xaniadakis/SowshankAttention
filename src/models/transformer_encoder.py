import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # even dims
        pe[:, 0::2] = torch.sin(position * div_term)
        # odd dims
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    # day_indices: (B, T)
    def forward(self, day_indices):
        # output shape: (B, T, d_model)
        # indexing on dim 0 (days)
        return self.pe[day_indices]

class TransformerTimeEncoder(nn.Module):
    def __init__(self, input_dim=128, num_heads=4, num_layers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=input_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x, day_of_year):
        # x: (B, T, F)
        # day_of_year: (B, T) ints (values 1–366)
        B, T, F = x.shape

        # positional encoding from actual dates
        pos = self.pos_encoder(day_of_year)  # (B, T, F)
        x = x + pos  # add temporal position encoding

        # prepend [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, F)
        x = torch.cat([cls_token, x], dim=1)  # (B, T+1, F)

        x = self.encoder(x)  # (B, T+1, F)
        x = x[:, 0]  # return only CLS token output → (B, F)
        x = self.dropout(x)
        x = self.bn(x)
        return x