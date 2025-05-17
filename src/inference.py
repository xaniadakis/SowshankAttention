import torch
from models.pixel_set_encoder import PixelSetEncoder
from models.transformer_encoder import TransformerTimeEncoder
from models.mlp import ClassifierHead, ParcelModel
from models.metrics import ClassificationMetrics
import json
import os
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
import numpy as np
from parcel_dataset import ParcelDataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_data_dir = os.path.join(base_dir, "data", "austria", "33UVP", "2017")
model_path = os.path.join(base_dir, "checkpoints", "best_model.pt")
meta_dir = os.path.join(base_dir, "data", "denmark", "32VNH", "2017", "meta")
test_meta_dir = os.path.join(base_dir, "data", "austria", "33UVP", "2017", "meta")
filtered_labels_path = os.path.join(test_meta_dir, "filtered_labels.json")

test_dataset = ParcelDataset(
    labels_json_path=filtered_labels_path,
    zarr_dir=os.path.join(test_data_dir, "data"),
    norm_stats_path=os.path.join(test_meta_dir, "normalization_stats.json"),
    train=False,
    sample_pixels=None,  # <- use all pixels
    # top_k=52
)

def variable_length_collate(batch):
    xs, ys, doys = zip(*batch)
    return list(xs), torch.tensor(ys), torch.stack(doys)

# 5. Get indices and labels for stratification on the filtered dataset
indices = np.arange(len(test_dataset))
with open(filtered_labels_path, "r") as f:
    filtered_labels_dict = json.load(f)

labels_array = np.array([filtered_labels_dict[str(test_dataset.parcel_ids[idx])] for idx in indices])

# Now you can do stratified sampling on indices + labels_array
fraction = 1.0
if fraction == 1.0:
    subset_idx = indices  # no splitting needed, use all indices
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-fraction, random_state=42)
    subset_idx, _ = next(sss.split(indices, labels_array))

# Use subset_idx with Subset and DataLoader for your inference
test_subset = Subset(test_dataset, subset_idx)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, collate_fn=variable_length_collate)

encoder = PixelSetEncoder(in_channels=12, hidden_dim=64, out_dim=128)
transformer = TransformerTimeEncoder(input_dim=128)
classifier = ClassifierHead(input_dim=128, num_classes=8)
model = ParcelModel(encoder, transformer, classifier).to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for xs, ys, doys in tqdm(test_loader):
        batch_preds = []
        for x, doy in zip(xs, doys):
            x = x.to(device)
            doy = doy.to(device)
            # The train dataset (Denmark) and test dataset (Austria) have different temporal lengths
            # model was trained with sequences of length 52, but during inference, it's getting sequences of length 58.
            # x, doy = pad_or_truncate(x, doy)
            x = x.unsqueeze(0)
            doy = doy.unsqueeze(0)
            output = model(x, doy)
            pred = torch.argmax(output, dim=1)
            batch_preds.append(pred.item())
        all_preds.extend(batch_preds)
        all_labels.extend(ys.cpu().numpy())

metrics = ClassificationMetrics(num_classes=8, device=device)
metrics.reset()
metrics.update(torch.tensor(all_preds).to(device), torch.tensor(all_labels).to(device))
results = metrics.compute()

print("Stratified inference on fraction of test dataset:")
for k, v in results.items():
    if k != "confusion_matrix":
        print(f"{k}: {v:.4f}")

# Convert predictions and labels to numpy
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

# Print classification report
print(classification_report(y_true, y_pred))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Class labels
class_names = [
    "unknown", "spring barley", "meadow", "winter wheat",
    "winter barley", "winter rye", "winter rapeseed", "corn"
]
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')

# Rotate the tick labels
plt.xticks(rotation=30, ha='right', fontsize=12)  # Rotate x-axis labels 45 degrees, right-aligned
plt.yticks(rotation=30, va='top', fontsize=12)    # Rotate y-axis labels 45 degrees, top-aligned

plt.tight_layout()
plt.savefig("./inference_confusion_matrix.png")