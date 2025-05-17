import geopandas as gpd
import os
import pickle
import json

region = "austria"
tile = "33UVP"
year = "2017"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
META_DIR = os.path.join(BASE_DIR, "..", "data", region, tile, year, "meta")
SRC_DIR = os.path.join(BASE_DIR)
MAPPING_PATH = os.path.join(SRC_DIR, "mapping.json")
LABELS_PATH = os.path.join(META_DIR, "labels.json")
FILTERED_LABELS_PATH = os.path.join(META_DIR, "filtered_labels.json")
PARCELS_PATH = os.path.join(META_DIR, "parcels", f"parcels_{region}_{tile}_{year}.shp")
META_PATH = os.path.join(META_DIR, "metadata.pkl")
DATES_JSON_PATH = os.path.join(META_DIR, "dates.json")

# Define desired final class names and their numeric IDs
class_to_index = {
    "unknown": 0,
    "spring_barley": 1,
    "meadow": 2,
    "winter_wheat": 3,
    "winter_barley": 4,
    "winter_rye": 5,
    "winter_rapeseed": 6,
    "corn": 7
}

if __name__ == "__main__":
    # Load the JSON mapping
    with open(MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)

    # Create a reverse mapping from German labels to English superlabels
    german_to_superlabel = {}
    for superlabel, entries in mapping_data.items():
        for entry in entries:
            german_to_superlabel[entry['german']] = superlabel

    # Read the parcels shapefile
    parcels = gpd.read_file(PARCELS_PATH)

    # Create labels dictionary with only known superlabels
    labels_dict = {}
    for idx, label in enumerate(parcels['snar_bezei']):
        superlabel = german_to_superlabel.get(label)
        if superlabel:
            labels_dict[str(idx)] = superlabel

    # Print distinct superlabels found
    print("Distinct superlabel categories found:")
    distinct_superlabels = set(labels_dict.values())
    for superlabel in sorted(distinct_superlabels):
        print(superlabel)

    # Save all mapped labels
    with open(LABELS_PATH, 'w') as f:
        json.dump(labels_dict, f, indent=2)

    # Filter the labels to include only those of interest
    filtered_labels_dict = {}
    for parcel_idx, superlabel in labels_dict.items():
        if superlabel in class_to_index:
            filtered_labels_dict[parcel_idx] = class_to_index[superlabel]

    # Print filtered class info
    print("\nDistinct filtered classes found (with indices):")
    found_classes = set(filtered_labels_dict.values())
    for idx in sorted(found_classes):
        for name, val in class_to_index.items():
            if val == idx:
                print(f"{name}: {idx}")

    # Save the filtered result
    with open(FILTERED_LABELS_PATH, 'w') as f:
        json.dump(filtered_labels_dict, f, indent=2)

    # Load metadata
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    # Extract and validate dates
    raw_dates = metadata.get("dates", [])
    print("Raw dates from metadata:", raw_dates)
    print("Unique dates count:", len(set(raw_dates)))

    if isinstance(raw_dates, (list, tuple)):
        valid_dates = [int(d) for d in raw_dates if isinstance(d, int)]
    else:
        valid_dates = [int(raw_dates)] if isinstance(raw_dates, int) else []

    # Save dates.json
    with open(DATES_JSON_PATH, "w") as f:
        json.dump(valid_dates, f, indent=2)

    print(f"Extracted {len(valid_dates)} dates to dates.json")