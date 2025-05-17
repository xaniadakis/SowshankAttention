import torch
import torch.nn as nn

class PixelSetEncoder(nn.Module):
    def __init__(self, in_channels=12, hidden_dim=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # attention mechanism
        # score each pixel
        self.attention = nn.Linear(out_dim, 1)  
        # normalize across pixels
        self.softmax = nn.Softmax(dim=2) 

    def forward(self, x):
        # x: (B, n_pixels, T, C)
        B, N, T, C = x.shape
        x = x.permute(0, 2, 1, 3)  # (B, T, n_pixels, C)
        x = x.reshape(B * T * N, C)
        x = self.mlp(x)  # (B*T*N, out_dim)
        x = x.view(B, T, N, -1)  # (B, T, n_pixels, out_dim)

        # compute attention scores
        attn_scores = self.attention(x)  # (B, T, n_pixels, 1)
        attn_weights = self.softmax(attn_scores)  # (B, T, n_pixels, 1)

        # weighted sum of pixel features
        x = (x * attn_weights).sum(dim=2)  # (B, T, out_dim)

        return x  # (B, T, out_dim)