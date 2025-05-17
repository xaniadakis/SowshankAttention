# SowshankAttention Project

## Overview
SowshankAttention is a geospatial machine learning project designed to process and analyze Sentinel-2 imagery for agricultural parcel classification. The project leverages attention-based models to capture temporal and spectral dynamics in multispectral data. It includes two primary pipelines: one for **training** a model on labeled parcel data and another for **inference** on test data. This README provides an overview of the project structure and instructions for executing both pipelines.

## Project Structure
The src/ directory houses the primary scripts and modules integral to the project’s functionality.
Here is a brief overview:

```
src/
├── category_preprocessing.py       # Preprocesses categorical data for training
├── normalization.py                # Normalizes multispectral data
├── parcel_dataset.py               # Dataset class for parcel data loading
├── models/                         # Model architectures and utilities
│   ├── metrics.py                  # Custom metrics for model evaluation
│   ├── pixel_set_encoder.py        # Pixel-set encoder for spatial features
│   ├── transformer_encoder.py      # Transformer-based encoder for temporal features
│   └── mlp.py                      # Multi-layer perceptron model
├── train.py                        # Trains the model
│
├── mapping.json                    # Mapping file for categorical labels
├── prepare_test_labels.py          # Prepares test labels for inference
└── inference.py                    # Runs inference on test data
```

The `data/` directory houses a subset or the entirety of the Timematch dataset, 
containing Sentinel-2 imagery, metadata and labels for agricultural parcels. 
It includes geospatial data in shapefile format, multispectral data in Zarr files, 
and metadata and label information in JSON and other relevant formats.
- A training dataset subset of Timematch, specifically for Denmark 2017 tile 32VNH, can be downloaded [here](https://drive.google.com/file/d/1h-eP8mWuqiHSs4XKgUYfNtT1kdwSfSeo/view)
- The full Timematch dataset is available at [Zenodo](https://zenodo.org/records/6542639)
```
data/
├── <region>/
    └── <tile>/
        └── <year>/
            ├── data/
            │   ├── 0.zarr/            
            │   │   ├── 0.0.0
            │   │   ├── 0.1.0
            │   │   ├── 1.0.0
            │   │   └── 1.1.0
            │   ├── 1.zarr/
            │   │   ├── 0.0.0
            │   │   └── 1.0.0
            │   └── 2.zarr/
            │       ├── 0.0.0
            │       └── 1.0.0
            └── meta/
                ├── blocks/
                │   ├── blocks_<region>_<tile>_<year>.cpg
                │   ├── blocks_<region>_<tile>_<year>.dbf
                │   ├── blocks_<region>_<tile>_<year>.prj
                │   ├── blocks_<region>_<tile>_<year>.shp
                │   └── blocks_<region>_<tile>_<year>.shx
                ├── dates.json          # acquisition dates 
                ├── labels.json         # raw parcel labels
                ├── metadata.pkl        # metadata in pickle format
                └── parcels/            
                    │  # for test dataset: has parcel information to derive labels
                    ├── parcels_<region>_<tile>_<year>.cpg
                    ├── parcels_<region>_<tile>_<year>.dbf
                    ├── parcels_<region>_<tile>_<year>.prj
                    ├── parcels_<region>_<tile>_<year>.shp
                    └── parcels_<region>_<tile>_<year>.shx
```
## Pipelines

### Training Pipeline
The training pipeline preprocesses data, normalizes it and trains the model. 

- **Category Preprocessing**
   ```bash
     python src/category_preprocessing.py
     ```

-  **Normalization**

     ```bash
     python src/normalization.py
     ```

-  **Training**:
     ```bash
     python src/train.py
     ```

### Inference Pipeline
The inference pipeline prepares test labels, normalizes the data and performs inference. 

-  **Prepare Test Labels** to recover and format the test labels for inference, consistent with the training pipeline
     ```bash
     python src/prepare_test_labels.py
     ```

-  **Normalization** to normalize the test data, consistent with the training pipeline.
     ```bash
     python src/normalization.py
     ```

-  **Inference** to generate predictions on the test data.
     ```bash
     python src/inference.py
     ```

## Notes
- Ensure all scripts are executed from the project root directory or adjust paths accordingly.
- The `mapping.json` file is critical for the inference pipeline, to map the german labels to our target classes.
- The input data should be placed inside the `data/` directory at the project root, 
following the structure of the Timematch dataset. Specifically, the hierarchy 
is arranged by country, then tile, followed by year, with separate `data` and `meta` subdirectories contained within each year folder.
The data directories are expected to contain multispectral data stored as Zarr files, while the meta directories include 
metadata files such as shapefiles, labels, acquisition dates and many others.

