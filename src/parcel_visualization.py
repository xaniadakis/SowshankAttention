# import zarr
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import random
#
# # Path to the directory containing .zarr folders
# DATA_DIR = "data/denmark/32VNH/2017/data"
#
# # List all .zarr folders
# zarr_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if d.endswith('.zarr')]
#
# # Pick one at random
# parcel_path = random.choice(zarr_dirs)
# print(f"Selected parcel: {parcel_path}")
#
# # Load Zarr data
# zarr_array = zarr.open(parcel_path, mode='r')  # shape: (num_pixels, 52, 10)
#
# # Convert to NumPy for visualization
# data = np.array(zarr_array)
#
# # Visualize the first pixel’s NDVI over time
# def compute_ndvi(band4, band8):
#     ndvi = (band8 - band4) / (band8 + band4 + 1e-6)
#     return ndvi
#
# pixel_idx = 0
# red_band = data[pixel_idx, :, 3]
# nir_band = data[pixel_idx, :, 7]
# ndvi = compute_ndvi(red_band, nir_band)
#
# plt.figure(figsize=(8, 4))
# plt.plot(ndvi, label="NDVI")
# plt.title("NDVI Time Series for Pixel 0")
# plt.xlabel("Time step (1 to 52)")
# plt.ylabel("NDVI")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
import zarr
import numpy as np
import matplotlib.pyplot as plt

# Distinct, visually clear colors chosen for each spectral band
band_info = [
    ('Blue (B2)',      '#1f77b4'),  # muted blue
    ('Green (B3)',     '#2ca02c'),  # green
    ('Red (B4)',       '#d62728'),  # red
    ('Red Edge 1 (B5)', '#ff7f0e'), # orange
    ('Red Edge 2 (B6)', '#9467bd'), # purple
    ('Red Edge 3 (B7)', '#8c564b'), # brownish
    ('NIR (B8)',       '#e377c2'),  # pink
    ('NIR narrow (B8A)','#7f7f7f'), # gray
    ('SWIR 1 (B11)',   '#bcbd22'),  # olive green
    ('SWIR 2 (B12)',   '#17becf')   # cyan
]

# Load one parcel
z = zarr.open('data/denmark/32VNH/2017/data/0.zarr', mode='r')
print(z.shape)  # (52, 10, N)

pixel_idx = 0
ts = z[:, :, pixel_idx]  # (52, 10)

plt.figure(figsize=(10, 6))
for b, (name, color) in enumerate(band_info):
    plt.plot(ts[:, b], label=name, color=color)
plt.xlabel('Time step (weeks)')
plt.ylabel('Reflectance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./spectral_time_series_pixel0_colored_distinct.png")

# /////////////////////////////////////////////////////////////////////////////////////
import pickle

# Load parcel metadata
with open('data/denmark/32VNH/2017/meta/metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

print(list(meta.keys())[:10])  # Print first 10 keys to check format
for i, parcel in enumerate(meta['parcels'][:3]):  # show first 3 entries
    print(f"Parcel {i} keys:", parcel.keys())

parcel_idx = 0
parcel = meta['parcels'][parcel_idx]

print(f"Parcel {parcel_idx} keys and sample values:\n")
for key, value in parcel.items():
    print(f"- {key} (type: {type(value)})")
    # Preview small data structures (dicts, lists, arrays)
    if isinstance(value, (list, dict)):
        print(f"  Sample: {list(value)[:3] if hasattr(value, '__iter__') else value}")
    elif hasattr(value, 'shape'):
        print(f"  shape: {value.shape}")
    else:
        print(f"  value: {value}")

import geopandas as gpd
import matplotlib.pyplot as plt

# Path to shapefile
shp_path = "data/denmark/32VNH/2017/meta/blocks/blocks_denmark_32VNH_2017.shp"

# Load shapefile
gdf = gpd.read_file(shp_path)

print(gdf.head())  # Inspect parcel polygons and attributes

# Plot all parcels
gdf.plot(edgecolor='black', facecolor='none')
plt.title('Parcel polygons - Denmark 32VNH 2017')
plt.show()


import zarr
import numpy as np
import matplotlib.pyplot as plt

# Load one parcel (e.g., 0.zarr)
zarr_path = "data/denmark/32VNH/2017/data/0.zarr"
z = zarr.open(zarr_path, mode='r')[:]  # shape: (52, 10, N)

# Pick one time step and band
time_step = 25
band_idx = 3
frame = z[time_step, band_idx, :]  # shape: (N,)

# Estimate spatial shape (e.g., square layout)
N = frame.shape[0]
side = int(np.sqrt(N))
while N % side != 0:
    side -= 1
H, W = side, N // side

# Reconstruct fake layout for visualization
img = frame.reshape(H, W)

# Plot
plt.imshow(img, cmap='viridis')
plt.colorbar(label='Reflectance')
plt.title(f"Parcel 0 – Band {band_idx} – Week {time_step}")
plt.axis('off')
plt.tight_layout()
plt.show()

# parcel_id = '0'
# shape = meta[parcel_id]['shape']  # (H, W)
# indices = meta[parcel_id]['indices']  # list of (row, col) for each pixel
#
# # Reconstruct image for time step 25 and band 3
# img = np.full(shape, np.nan)
# for i, (r, c) in enumerate(indices):
#     img[r, c] = z[25, 3, i]
#
# plt.imshow(img, cmap='viridis')
# plt.colorbar(label='Reflectance')
# plt.title('Band 3 at week 25')
# plt.axis('off')
# plt.show()
# /////////////////////////////////////////////////////////////////////////////////////
# mean_ts = z[:, :, :].mean(axis=2)  # shape: (52, 10)
#
# plt.figure(figsize=(10, 6))
# for b in range(mean_ts.shape[1]):
#     plt.plot(mean_ts[:, b], label=f'Band {b}')
# plt.xlabel('Time step')
# plt.ylabel('Mean Reflectance')
# plt.title('Mean spectral reflectance over time (parcel average)')
# plt.legend()
# plt.grid(True)
# plt.show()
