import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm, trange

from parcel_dataset import ParcelDataset
from models.pixel_set_encoder import PixelSetEncoder
from models.transformer_encoder import TransformerTimeEncoder
from models.mlp import ClassifierHead, ParcelModel
from models.metrics import ClassificationMetrics

def color(text, style='bold', color='cyan'):
    styles = {'bold': '1', 'dim': '2', 'normal': '22'}
    colors = {'cyan': '36', 'magenta': '35', 'yellow': '33', 'blue': '34'}
    style_code = styles.get(style, '22')
    color_code = colors.get(color, '36')
    return f"\033[{style_code};{color_code}m{text}\033[0m"

parser = argparse.ArgumentParser(description="Train ParcelModel with different modes")
parser.add_argument('--mode', type=str, default='split', choices=['split', 'kfold'],
                    help="Training mode: 'split' for stratified train-test split, 'kfold' for K-fold cross-validation")
args = parser.parse_args()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "denmark", "32VNH", "2017")

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k_folds = 5
epochs = 100
batch_size = 64
lr = 1e-4
patience = 10  # early stopping patience
model_out_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(model_out_dir, exist_ok=True)

print(f"We are running on {device}!")

# dataset
dataset = ParcelDataset(
    labels_json_path=os.path.join(data_dir, "meta", "filtered_labels.json"),
    zarr_dir=os.path.join(data_dir, "data"),
    norm_stats_path=os.path.join(data_dir, "meta", "normalization_stats.json"),
    train=True,
    sample_pixels=batch_size
)

# load labels for stratification
with open(os.path.join(data_dir, "meta", "filtered_labels.json"), "r") as f:
    labels_dict = json.load(f)
indices = np.arange(len(dataset))
labels = np.array([int(labels_dict[dataset.parcel_ids[idx]]) for idx in indices])

# initialize metrics tracking
all_metrics = []

if args.mode == 'kfold':
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n🌀 Fold {fold + 1}/{k_folds}")

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=False)

        # initialize our model
        encoder = PixelSetEncoder(in_channels=12, hidden_dim=64, out_dim=128)
        transformer = TransformerTimeEncoder(input_dim=128)
        classifier = ClassifierHead(input_dim=128, num_classes=8)
        model = ParcelModel(encoder, transformer, classifier).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # load class counts from filtered_labels.json
        label_counts = np.bincount([int(v) for v in labels_dict.values()])
        class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum() * len(label_counts)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        metrics = ClassificationMetrics(num_classes=8, device=device)

        best_val_f1 = 0
        bad_epochs = 0
        train_losses, val_f1s, y_true, y_pred = [], [], [], []

        for epoch in range(epochs):
            # training phase
            model.train()
            train_loss = 0
            for x, y, doy in train_loader:
                x, y, doy = x.to(device), y.to(device), doy.to(device)
                optimizer.zero_grad()
                outputs = model(x, doy)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(train_loader))

            # validation phase
            model.eval()
            metrics.reset()
            with torch.no_grad():
                for x, y, doy in val_loader:
                    x, y, doy = x.to(device), y.to(device), doy.to(device)
                    out = model(x, doy)
                    preds = torch.argmax(out, dim=1)
                    metrics.update(preds, y)
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            val_result = metrics.compute()
            val_f1 = val_result["f1_micro"]
            val_f1s.append(val_f1)
            scheduler.step(val_f1)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val F1_micro: {val_f1:.4f}")

            # early stopping w/ some checkpointing
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                bad_epochs = 0
                ckpt_path = os.path.join(model_out_dir, f"best_model_fold{fold+1}.pt")
                torch.save(model.state_dict(), ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                    break

        # final evaluation with best model
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        metrics.reset()
        with torch.no_grad():
            for x, y, doy in val_loader:
                x, y, doy = x.to(device), y.to(device), doy.to(device)
                out = model(x, doy)
                preds = torch.argmax(out, dim=1)
                metrics.update(preds, y)

        result = metrics.compute()
        all_metrics.append(result)
        print("✅ Fold Final Metrics:")
        for k, v in result.items():
            if k != "confusion_matrix":
                print(f"{k}: {v:.4f}")

        # visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_f1s, label="Validation F1-micro")
        plt.xlabel("Epoch")
        plt.ylabel("F1-micro")
        plt.legend()
        plt.savefig(os.path.join(model_out_dir, f"training_curves_fold{fold+1}.png"))
        plt.close()

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(model_out_dir, f"confusion_matrix_fold{fold+1}.png"))
        plt.close()

    # mean/std reporting for K-fold run
    print("\n📊 Cross-Validation Summary:")
    keys = ["accuracy", "f1_micro", "f1_weighted", "precision", "recall"]
    for key in keys:
        vals = [m[key] for m in all_metrics]
        print(f"{key}: mean = {np.mean(vals):.4f}, std = {np.std(vals):.4f}")

else:
    # stratified train-test split
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=False)

    # init our model
    encoder = PixelSetEncoder(in_channels=12, hidden_dim=64, out_dim=128)
    transformer = TransformerTimeEncoder(input_dim=128)
    classifier = ClassifierHead(input_dim=128, num_classes=8)
    model = ParcelModel(encoder, transformer, classifier).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # load class counts from filtered_labels.json
    label_counts = np.bincount([int(v) for v in labels_dict.values()])
    class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(label_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    metrics = ClassificationMetrics(num_classes=8, device=device)

    best_val_f1 = 0
    bad_epochs = 0
    train_losses, val_f1s, y_true, y_pred = [], [], [], []

    epoch_bar = trange(epochs, desc="Epochs", colour="red")
    for epoch in epoch_bar:
        # training phase
        model.train()
        metrics.reset()
        train_loss = 0
        for x, y, doy in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", leave=False):
            x, y, doy = x.to(device), y.to(device), doy.to(device)
            optimizer.zero_grad()
            outputs = model(x, doy)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, y)

        train_losses.append(train_loss / len(train_loader))
        train_result = metrics.compute()
        train_f1 = train_result["f1_weighted"]
        train_acc = train_result["accuracy"]

        # evaluation phase
        model.eval()
        metrics.reset()
        val_loss = 0
        with torch.no_grad():
            for x, y, doy in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}", leave=False):
                x, y, doy = x.to(device), y.to(device), doy.to(device)
                out = model(x, doy)
                loss = criterion(out, y)
                val_loss += loss.item()

                preds = torch.argmax(out, dim=1)
                metrics.update(preds, y)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_result = metrics.compute()
        val_f1 = val_result["f1_weighted"]
        val_acc = val_result["accuracy"]
        val_f1s.append(val_f1)
        scheduler.step(val_f1)

        tqdm.write(
            f"{color('Epoch', 'bold', 'magenta')} {epoch + 1}/{epochs} | "
            f"{color('Train', 'bold', 'cyan')}: "
            f"{color('loss', color='cyan')} {train_loss:.4f} "
            f"{color('f1', color='cyan')} {train_f1:.4f} "
            f"{color('acc', color='cyan')} {train_acc:.2%} | "
            f"{color('Val', 'bold', 'blue')}: "
            f"{color('loss', color='blue')} {val_loss:.4f} "
            f"{color('f1', color='blue')} {val_f1:.4f} "
            f"{color('acc', color='blue')} {val_acc:.2%}"
        )

        # early stopping w/ some of checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            bad_epochs = 0
            ckpt_path = os.path.join(model_out_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    # final evaluation with best model
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for x, y, doy in val_loader:
            x, y, doy = x.to(device), y.to(device), doy.to(device)
            out = model(x, doy)
            preds = torch.argmax(out, dim=1)
            metrics.update(preds, y)

    result = metrics.compute()
    print("✅ Final Metrics:")
    for k, v in result.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, label="Validation F1-micro")
    plt.xlabel("Epoch")
    plt.ylabel("F1-micro")
    plt.legend()
    plt.savefig(os.path.join(model_out_dir, "training_curves.png"))
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(model_out_dir, "confusion_matrix.png"))
    plt.close()