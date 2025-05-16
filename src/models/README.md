# Parcel Classification Model – Overview

This repository implements a deep learning model for classifying parcels (e.g., agricultural fields) based on time-series satellite imagery. The model processes unordered sets of pixel-level observations over time, capturing both spatial and temporal patterns to predict a parcel's class (e.g., crop type).

The architecture consists of three main components:
1. **Pixel Set Encoder**: Encodes spatial information from sets of pixels at each timestep.
2. **Transformer Time Encoder**: Models temporal dynamics across the sequence of encoded timesteps.
3. **Classifier Head**: Maps the final parcel representation to class probabilities.
4. **Parcel Model**: Combines the above components into a single end-to-end model.

The model is inspired by remote sensing literature (Pelletier et al., 2019) and Transformer architectures (Vaswani et al., 2017).

---

## Pixel Set Encoder – Core Idea (from Pelletier et al., 2019)

The **Pixel Set Encoder** processes spatially unordered sets of pixel-level observations for each timestep, typical in remote sensing data.

### Goal
Transform a set of pixels per timestep (e.g., 32 pixels for day `t`) into a single feature vector that captures the spatial structure of the parcel at that time.

### Input
- Shape: `(B, N_pixels, T, C)`
  - `B`: Batch size
  - `N_pixels`: Number of pixels per parcel (e.g., 32)
  - `T`: Number of timesteps (e.g., 52)
  - `C`: Number of channels (e.g., 12, such as spectral bands)

### Steps
1. **Shared MLP per Pixel**:
   - A Multi-Layer Perceptron (MLP) processes each pixel vector (shape: `[C]`).
   - Architecture: `Linear(C, hidden_dim) → ReLU → Linear(hidden_dim, out_dim)`
   - Default parameters: `in_channels=12`, `hidden_dim=64`, `out_dim=128`
   - Output: Each pixel is transformed into a latent feature vector of size `out_dim`.

2. **Attention-Based Aggregation**:
   - An attention mechanism computes scores for each pixel using a linear layer (`Linear(out_dim, 1)`).
   - Scores are normalized across pixels using `Softmax` to obtain attention weights.
   - A weighted sum of pixel features is computed per timestep, producing a single vector.

3. **Final Output**:
   - Shape: `(B, T, out_dim)` (e.g., `(B, T, 128)`)
   - Each vector represents the spatial information of the parcel at a specific timestep.

### Benefits
- Handles variable-sized pixel sets (permutation invariance).
- Attention mechanism prioritizes informative pixels.
- Bridges raw spatial data to temporal modeling.

### Paper Reference
Pelletier et al., 2019 — "Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series"  
[https://arxiv.org/abs/1911.07757](https://arxiv.org/abs/1911.07757)

---

## Transformer Time Encoder – Core Idea

The **Transformer Time Encoder** models temporal dynamics across the sequence of encoded timestep features, producing a single parcel-level representation.

### Goal
Use self-attention to capture long-range temporal dependencies and aggregate information into a single vector per parcel.

### Input
- **Features**: Shape `(B, T, F)` (e.g., `(B, T, 128)`), output from the Pixel Set Encoder.
- **Day of Year**: Shape `(B, T)`, integer values (1–366) indicating the day of observation.

### Steps
1. **Sinusoidal Positional Encoding**:
   - A fixed positional encoding is computed based on the day-of-year indices using sine and cosine functions.
   - Parameters: `d_model=F` (e.g., 128), `max_len=366` (days in a year).
   - The encoding is added to the input features to provide temporal context.

2. **[CLS] Token Insertion**:
   - A learnable `[CLS]` token (shape: `(1, 1, F)`) is prepended to the sequence.
   - After attention, the `[CLS]` token’s output represents the entire parcel.

3. **Transformer Layers**:
   - A stack of Transformer encoder layers processes the sequence.
   - Parameters: `input_dim=128`, `num_heads=4`, `num_layers=2`, `ff_dim=256`, `dropout=0.1`.
   - Uses batch-first format and multi-head self-attention.

4. **Post-Processing**:
   - The output of the `[CLS]` token is extracted (shape: `(B, F)`).
   - Dropout (`0.1`) and batch normalization are applied to stabilize training.

5. **Final Output**:
   - Shape: `(B, F)` (e.g., `(B, 128)`), a single vector per parcel.

### Benefits
- Captures long-range temporal dependencies.
- Learns which timesteps are most relevant via attention.
- Flexible to irregular time intervals using day-of-year encodings.

### Paper Reference
Vaswani et al., 2017 — "Attention is All You Need"  
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

## Classifier Head – Core Idea

The **Classifier Head** maps the parcel-level representation to class probabilities for the final prediction.

### Goal
Transform the Transformer’s output vector into a probability distribution over possible classes.

### Input
- Shape: `(B, F)` (e.g., `(B, 128)`), output from the Transformer Time Encoder.

### Steps
1. **MLP Architecture**:
   - A sequential neural network processes the input:
     - `Linear(input_dim, hidden_dim)` → `BatchNorm1d` → `ReLU` → `Dropout` → `Linear(hidden_dim, num_classes)`
   - Default parameters: `input_dim=128`, `hidden_dim=64`, `num_classes=8`, `dropout=0.3`.

2. **Final Output**:
   - Shape: `(B, num_classes)` (e.g., `(B, 8)`), raw logits for each class.
   - Typically followed by a `Softmax` or `CrossEntropyLoss` during training/inference.

### Benefits
- Simple yet effective for classification.
- Dropout and batch normalization improve generalization.
- Flexible to different numbers of classes.

---

## Parcel Model – Core Idea

The **Parcel Model** integrates the Pixel Set Encoder, Transformer Time Encoder, and Classifier Head into a single end-to-end architecture.

### Goal
Process raw pixel-level time-series data and predict the class of each parcel.

### Input
- **Pixel Data**: Shape `(B, N_pixels, T, C)` (e.g., `(B, 32, 52, 12)`).
- **Day of Year**: Shape `(B, T)`, integer values (1–366).

### Steps
1. **Pixel Set Encoder**:
   - Transforms `(B, N_pixels, T, C)` → `(B, T, F)` using MLP and attention-based aggregation.

2. **Transformer Time Encoder**:
   - Processes `(B, T, F)` with day-of-year encodings → `(B, F)` using self-attention and `[CLS]` token.

3. **Classifier Head**:
   - Maps `(B, F)` → `(B, num_classes)` to produce class logits.

### Final Output
- Shape: `(B, num_classes)` (e.g., `(B, 8)`), class logits for each parcel.

### Benefits
- End-to-end training for spatial and temporal feature learning.
- Modular design allows easy replacement of components.
- Robust to variable pixel counts and irregular time series.

---

## Implementation Details

- **Code Files**:
  - `pixel_set_encoder.py`: Implements the `PixelSetEncoder` class.
  - `transformer_encoder.py`: Implements the `SinusoidalPositionalEncoding` and `TransformerTimeEncoder` classes.
  - `mlp.py`: Implements the `ClassifierHead` and `ParcelModel` classes.

- **Dependencies**:
  - PyTorch (`torch`, `torch.nn`)
  - Math (for positional encoding)

- **Default Hyperparameters**:
  - Pixel Set Encoder: `in_channels=12`, `hidden_dim=64`, `out_dim=128`
  - Transformer Time Encoder: `input_dim=128`, `num_heads=4`, `num_layers=2`, `ff_dim=256`, `dropout=0.1`
  - Classifier Head: `input_dim=128`, `hidden_dim=64`, `num_classes=8`, `dropout=0.3`

- **Training Considerations**:
  - Use `CrossEntropyLoss` for multi-class classification.
  - Apply `Softmax` on logits for inference.
  - Batch normalization and dropout help prevent overfitting.

---

## Usage Example

```python
import torch
from pixel_set_encoder import PixelSetEncoder
from transformer_encoder import TransformerTimeEncoder
from mlp import ClassifierHead, ParcelModel

# Initialize components
encoder = PixelSetEncoder(in_channels=12, hidden_dim=64, out_dim=128)
transformer = TransformerTimeEncoder(input_dim=128, num_heads=4, num_layers=2, ff_dim=256, dropout=0.1)
classifier = ClassifierHead(input_dim=128, hidden_dim=64, num_classes=8, dropout=0.3)

# Combine into ParcelModel
model = ParcelModel(encoder, transformer, classifier)

# Example input
x = torch.randn(16, 32, 52, 12)  # (B, N_pixels, T, C)
day_of_year = torch.randint(1, 367, (16, 52))  # (B, T)

# Forward pass
logits = model(x, day_of_year)  # (B, num_classes)
print(logits.shape)  # torch.Size([16, 8])