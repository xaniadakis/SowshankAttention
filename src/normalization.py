import os
import json
import pickle
import zarr
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], required=True)
args = parser.parse_args()
mode = args.mode

if mode == "train":
    region = "denmark"
    tile = "32VNH"
    year = "2017"
    expected_timesteps = 52
elif mode == "test":
    region = "austria"
    tile = "33UVP"
    year = "2017"
    expected_timesteps = 58

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", region, tile, year, "data")
meta_dir = os.path.join(base_dir, "data", region, tile, year, "meta")
labels_path = os.path.join(meta_dir, "filtered_labels.json")
meta_path = os.path.join(meta_dir, "metadata.pkl")
norm_stats_path = os.path.join(meta_dir, "normalization_stats.json")

with open(labels_path, "r") as f:
    labels = json.load(f)
label_keys = list(labels.keys())

# Subsample to run faster
# if mode == "test":
#     label_keys = label_keys[::100]

# load metadata for cloud filtering
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
cloudy_pct = np.array(meta["cloudy_pct"])
cloud_threshold = np.percentile(cloudy_pct, 75)

print(f"[{mode.upper()}] Cloud threshold: {cloud_threshold:.2f}%")
print(f"[{mode.upper()}] Valid time steps: {(cloudy_pct <= cloud_threshold).sum()} / {len(cloudy_pct)}")

channel_mean = np.zeros(12)
channel_M2 = np.zeros(12)
count = 0

for parcel_id in tqdm(label_keys, desc=f"Computing normalization ({mode})"):
    parcel_path = os.path.join(data_dir, f"{parcel_id}.zarr")
    if not os.path.exists(parcel_path):
        continue
    try:
        data = zarr.open(parcel_path, mode="r")[:]
        if data.shape[0] == expected_timesteps and data.shape[1] == 10:
            valid_mask = cloudy_pct <= cloud_threshold
            if valid_mask.sum() == 0:
                continue
            data = data[valid_mask]
            data = np.transpose(data, (2, 0, 1))
        else:
            continue

        nir = data[:, :, 6]
        red = data[:, :, 2]
        blue = data[:, :, 0]
        ndvi = (nir - red) / (nir + red + 1e-10)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)
        data = np.concatenate([data, ndvi[:, :, None], evi[:, :, None]], axis=2)
        flat = data.reshape(-1, 12)

        for x in flat:
            count += 1
            delta = x - channel_mean
            channel_mean += delta / count
            channel_M2 += delta * (x - channel_mean)
    except Exception as e:
        print(f"Skipping {parcel_id}: {e}")

mean = channel_mean
std = np.sqrt(channel_M2 / (count - 1)) if count > 1 else np.zeros(12)

print(f"\n[{mode.upper()}] Channel-wise mean:\n", mean)
print(f"\n[{mode.upper()}] Channel-wise std:\n", std)

with open(norm_stats_path, "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

print(f"\nSaved normalization stats to: {norm_stats_path}")
