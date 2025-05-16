import json
import os
from collections import Counter

MIN_SAMPLES = 200

def load_labels(labels_path):
    with open(labels_path, "r") as f:
        return json.load(f)

def count_categories(labels):
    return Counter(labels.values())

def get_new_indexing(label_counts):
    sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])
    return {label: idx for idx, (label, _) in enumerate(sorted_labels)}

def get_custom_indexing(label_counts, prioritize_label=None):
    labels = list(label_counts.keys())

    if prioritize_label and prioritize_label in label_counts:
        labels.remove(prioritize_label)
        sorted_labels = [prioritize_label] + sorted(labels, key=lambda x: -label_counts[x])
    else:
        sorted_labels = sorted(labels, key=lambda x: -label_counts[x])

    return {label: idx for idx, label in enumerate(sorted_labels)}

def filter_labels(label_counts, min_samples):
    return {label: count for label, count in label_counts.items() if count >= min_samples}

def remap_parcels(labels, filtered_label_to_index):
    return {
        parcel_id: filtered_label_to_index[label]
        for parcel_id, label in labels.items()
        if label in filtered_label_to_index
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labels_path = os.path.join(base_dir, "data", "denmark", "32VNH", "2017", "meta", "labels.json")

    labels = load_labels(labels_path)
    label_counts = count_categories(labels)

    print("Category counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    label_to_index = get_new_indexing(label_counts)
    print("\nNew label indexing:")
    print(label_to_index)

    filtered_label_counts = filter_labels(label_counts, MIN_SAMPLES)
    filtered_label_to_index = get_custom_indexing(filtered_label_counts, prioritize_label="unknown")

    print("\nFiltered categories (≥200 samples):")
    for label, count in filtered_label_counts.items():
        print(f"{label}: {count}")

    print("\nFiltered label indexing:")
    print(filtered_label_to_index)

    remapped_labels = remap_parcels(labels, filtered_label_to_index)
    print(f"\nTotal valid parcels after filtering: {len(remapped_labels)}")

    filtered_output_path = os.path.join(os.path.dirname(labels_path), "filtered_labels.json")
    with open(filtered_output_path, "w") as f:
        json.dump(remapped_labels, f, indent=2)

    print(f"Filtered + indexed labels saved to: {filtered_output_path}")
