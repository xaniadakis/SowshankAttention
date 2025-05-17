import os
import json
import pickle
from datetime import datetime
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans

class ParcelDataset(Dataset):
    def __init__(self, labels_json_path, zarr_dir, norm_stats_path, train=True, sample_pixels=32, top_k=None):
        with open(labels_json_path, "r") as f:
            self.label_dict = json.load(f)

        with open(norm_stats_path, "r") as f:
            stats = json.load(f)
            self.mean = np.array(stats["mean"])
            self.std = np.array(stats["std"])

        self.zarr_dir = zarr_dir
        self.parcel_ids = [
            pid for pid in self.label_dict.keys()
            if os.path.exists(os.path.join(self.zarr_dir, f"{pid}.zarr"))
        ]
        self.train = train
        self.sample_pixels = sample_pixels
        self.top_k = top_k

        # load metadata from metadata.pkl
        meta_path = os.path.join(os.path.dirname(labels_json_path), "metadata.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.dates = meta["dates"]
        self.cloudy_pct = np.array(meta["cloudy_pct"])
        self.day_of_year = self._convert_to_day_of_year(self.dates)

        # dynamic threshold based on 75th percentile
        # self.cloud_threshold = np.percentile(self.cloudy_pct, 75)
        # print(f"Dynamic cloud threshold set to: {self.cloud_threshold:.2f}%")

    def __len__(self):
        return len(self.parcel_ids)

    def __getitem__(self, idx):
        parcel_id = self.parcel_ids[idx]
        label = self.label_dict[parcel_id]
        zarr_path = os.path.join(self.zarr_dir, f"{parcel_id}.zarr")
        data = zarr.open(zarr_path, mode="r")[:]  # (52, 10, N)

        # # filter time steps based on dynamic cloud percentage
        # valid_mask = self.cloudy_pct <= self.cloud_threshold
        # if valid_mask.sum() < 5:  # Minimum 5 time steps required
        #     relaxed_threshold = self.cloud_threshold + 10
        #     valid_mask = self.cloudy_pct <= relaxed_threshold
        #     if valid_mask.sum() == 0:
        #         raise ValueError(f"No valid time steps for parcel {parcel_id} even with relaxed threshold")
        # data = data[valid_mask]  # (valid_timesteps, 10, N)

        # transpose to (N, valid_timesteps, 10)
        if data.shape[1] == 10:
            data = np.transpose(data, (2, 0, 1))
        else:
            raise ValueError(f"Invalid shape for parcel {parcel_id} after filtering: {data.shape}")

        # # normalize
        # data = (data - self.mean) / self.std

        # interpolate to restore 52 time steps
        # if data.shape[1] < 52:
        #     full_data = np.zeros((data.shape[0], 52, data.shape[2]))
        #     valid_indices = np.where(valid_mask)[0]
        #     for pixel in range(data.shape[0]):
        #         for band in range(data.shape[2]):
        #             valid_data = data[pixel, :, band]
        #             if len(valid_data) > 1:
        #                 spline = CubicSpline(valid_indices, valid_data)
        #                 full_data[pixel, :, band] = spline(np.arange(52))
        #             else:
        #                 full_data[pixel, :, band] = np.interp(np.arange(52), valid_indices, valid_data)
        #     data = full_data
        # else:
        #     data = data[:, :52, :]  # Ensure 52 time steps

        # if self.train:
        #     n_pixels = data.shape[0]
        #     if n_pixels > self.sample_pixels:
        #         # Cluster pixels based on mean spectral features
        #         mean_spectral = data.mean(axis=1)  # Shape: (N, 10)
        #         kmeans = KMeans(n_clusters=3, random_state=42)
        #         clusters = kmeans.fit_predict(mean_spectral)
        #         indices = []
        #         for cluster in range(3):
        #             cluster_indices = np.where(clusters == cluster)[0]
        #             cluster_size = len(cluster_indices)
        #             sample_size = int(self.sample_pixels * (cluster_size / n_pixels))
        #             sample_size = min(sample_size, cluster_size)
        #             indices.extend(np.random.choice(cluster_indices, size=sample_size, replace=False))
        #         if len(indices) < self.sample_pixels:
        #             remaining = self.sample_pixels - len(indices)
        #             other_indices = np.setdiff1d(np.arange(n_pixels), indices)
        #             indices.extend(np.random.choice(other_indices, size=remaining, replace=False))
        #         data = data[indices]
        #     else:
        #         data = data

        # n_pixels = data.shape[0]
        # if self.train and n_pixels > self.sample_pixels:
        #     def fps(X, k):
        #         selected = [np.random.randint(len(X))]
        #         for _ in range(k - 1):
        #             dists = np.linalg.norm(X - X[selected][:, None], axis=2).min(axis=0)
        #             selected.append(np.argmax(dists))
        #         return selected
        #
        #     spectral_mean = data.mean(axis=1)  # (N, 10)
        #     indices = fps(spectral_mean, self.sample_pixels)
        #     data = data[indices]
        # elif self.train:
        #     indices = np.random.choice(n_pixels, size=self.sample_pixels, replace=n_pixels < self.sample_pixels)
        #     data = data[indices]

        # if top_k is specified, filter time steps
        if self.top_k is not None:
            # get indices of top_k least cloudy time steps
            topk_cloudy_indices = np.argsort(self.cloudy_pct)[:self.top_k]
            # sort these indices by date (day of year)
            topk_ordered = sorted(topk_cloudy_indices, key=lambda i: self.day_of_year[i])

            print(f"data shape: {data.shape}")
            print(f"cloudy_pct length: {len(self.cloudy_pct)}")
            print(f"topk indices: {topk_cloudy_indices}")
            print(f"max index in topk_ordered: {max(topk_ordered)}")

            # keep only the selected time steps in data (N pixels, top_k times, 10 bands)
            data = data[:, topk_ordered, :]
            # extract the day of year values for the selected time steps
            doy = [self.day_of_year[i] for i in topk_ordered]

        # random pixel sampling
        if self.train:
            n_pixels = data.shape[0]
            indices = np.random.choice(n_pixels, size=self.sample_pixels, replace=n_pixels < self.sample_pixels)
            data = data[indices]

        # compute NDVI and EVI
        nir = data[:, :, 6]  # B8 (NIR)
        red = data[:, :, 2]  # B4 (Red)
        blue = data[:, :, 0]  # B2 (Blue)
        # avoid division by zero
        ndvi = (nir - red) / (nir + red + 1e-10)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)
        data = np.concatenate([data, ndvi[:, :, np.newaxis], evi[:, :, np.newaxis]], axis=2)  # (N, 52, 12)

        # augment & normalize
        if self.train:
            data = self._augment(data)
        data = (data - self.mean) / self.std

        doy_tensor = torch.tensor(doy, dtype=torch.long) if self.top_k is not None else torch.tensor(self.day_of_year, dtype=torch.long)
        return (
            torch.tensor(data, dtype=torch.float32),  # (sample_pixels, top_k, 12)
            torch.tensor(label, dtype=torch.long),
            doy_tensor
        )

    def _convert_to_day_of_year(self, date_list):
        return [datetime.strptime(str(d), "%Y%m%d").timetuple().tm_yday for d in date_list]

    def _augment(self, data):
        # data: (N, 52, 12)
        N, T, C = data.shape

        # spectral jitter (per band)
        if np.random.rand() < 0.8:
            noise = np.random.normal(0, 0.05, size=data.shape)
            data += noise

        # time shift
        if np.random.rand() < 0.5:
            shift = np.random.randint(-5, 6)
            data = np.roll(data, shift, axis=1)

        # random dropout of 1-2 bands
        if np.random.rand() < 0.3:
            drop_channels = np.random.choice(C, size=np.random.randint(1, 3), replace=False)
            data[:, :, drop_channels] = 0

        # gaussian blur in time (simulate atmospheric smoothing)
        if np.random.rand() < 0.3:
            from scipy.ndimage import gaussian_filter1d
            for i in range(N):
                data[i] = gaussian_filter1d(data[i], sigma=1.0, axis=0)

        return data

    def compute_cloud_stats(self, top_k=None):
        if top_k is not None:
            selected_indices = np.argsort(self.cloudy_pct)[:top_k]
            used_cloud = self.cloudy_pct[selected_indices]
        else:
            used_cloud = self.cloudy_pct

        valid_timesteps = (used_cloud <= self.cloud_threshold).sum()
        total_timesteps = len(used_cloud)
        valid_pct = (valid_timesteps / total_timesteps) * 100
        invalid_pct = 100 - valid_pct
        avg_cloud = np.mean(used_cloud)

        print(f"Cloud stats for top-{top_k if top_k else total_timesteps} time steps:")
        print(f"  - Valid time steps (below threshold): {valid_pct:.2f}%")
        print(f"  - Invalid time steps (above threshold): {invalid_pct:.2f}%")
        print(f"  - Average cloud cover: {avg_cloud:.2f}%")
        return valid_pct, invalid_pct, avg_cloud


if __name__ == '__main__':
    top_k = None
    dataset = ParcelDataset(
        labels_json_path="data/denmark/32VNH/2017/meta/filtered_labels.json",
        zarr_dir="data/denmark/32VNH/2017/data",
        norm_stats_path="data/denmark/32VNH/2017/meta/normalization_stats.json",
        train=True,
        sample_pixels=32,
        top_k=top_k
    )

    dataset.compute_cloud_stats(top_k=top_k)

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for x, y, doy in loader:
        print(x.shape)
        print(y.shape)
        print(doy.shape)
        break