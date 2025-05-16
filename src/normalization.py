import os
import json
import pickle
import zarr
import numpy as np
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "denmark", "32VNH", "2017", "data")
labels_path = os.path.join(base_dir, "data", "denmark", "32VNH", "2017", "meta", "filtered_labels.json")
meta_path = os.path.join(base_dir, "data", "denmark", "32VNH", "2017", "meta", "metadata.pkl")

# load labels
with open(labels_path, "r") as f:
    labels = json.load(f)

# load metadata for cloud filtering
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
cloudy_pct = np.array(meta["cloudy_pct"])
# dynamic threshold based on 75th percentile
cloud_threshold = np.percentile(cloudy_pct, 75)

# compute cloud coverage statistics
valid_timesteps = (cloudy_pct <= cloud_threshold).sum()
total_timesteps = len(cloudy_pct)
valid_pct = (valid_timesteps / total_timesteps) * 100
invalid_pct = 100 - valid_pct
avg_cloud = np.mean(cloudy_pct)
print(f"Dynamic cloud threshold set to: {cloud_threshold:.2f}%")
print(f"Percentage of valid time steps (below {cloud_threshold:.2f}% cloud cover): {valid_pct:.2f}%")
print(f"Percentage of invalid time steps (above {cloud_threshold:.2f}% cloud cover): {invalid_pct:.2f}%")
print(f"Average cloud coverage across all parcels: {avg_cloud:.2f}%")

# init running sum and squared sum for 12 channels (10 bands + NDVI + EVI)
channel_mean = np.zeros(12)
channel_M2 = np.zeros(12)
count = 0

for parcel_id in tqdm(labels.keys(), desc="Computing normalization"):
    parcel_path = os.path.join(data_dir, f"{parcel_id}.zarr")
    if not os.path.exists(parcel_path):
        continue
    try:
        data = zarr.open(parcel_path, mode="r")[:]
        if data.shape[0] == 52 and data.shape[1] == 10:
            # apply cloud filtering
            valid_mask = cloudy_pct <= cloud_threshold
            if valid_mask.sum() == 0:
                print(f"Skipping {parcel_id}: no valid time steps after cloud filtering")
                continue
            data = data[valid_mask]  # (valid_timesteps, 10, N)
            data = np.transpose(data, (2, 0, 1))  # (pixels, valid_timesteps, 10)
        else:
            print(f"Skipping {parcel_id}: unexpected shape {data.shape}")
            continue

        # compute NDVI and EVI
        nir = data[:, :, 6]  # B8 (NIR)
        red = data[:, :, 2]  # B4 (Red)
        blue = data[:, :, 0]  # B2 (Blue)
        ndvi = (nir - red) / (nir + red + 1e-10)  # avoid division by zero
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)
        data = np.concatenate([data, ndvi[:, :, np.newaxis], evi[:, :, np.newaxis]], axis=2)  # (pixels, valid_timesteps, 12)

        # flatten for online mean and variance computation
        flat = data.reshape(-1, 12)  # (pixels * valid_timesteps, 12)

        for x in flat:
            count += 1
            delta = x - channel_mean
            channel_mean += delta / count
            channel_M2 += delta * (x - channel_mean)
    except Exception as e:
        print(f"Skipping {parcel_id}: {e}")

mean = channel_mean
var = channel_M2 / (count - 1) if count > 1 else np.zeros(12)
std = np.sqrt(var)

print("\nChannel-wise mean (10 bands + NDVI + EVI):\n", mean)
print("\nChannel-wise std (10 bands + NDVI + EVI):\n", std)

norm_stats_path = os.path.join(base_dir, "data", "normalization_stats.json")
with open(norm_stats_path, "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

print(f"\nSaved normalization stats to: {norm_stats_path}")