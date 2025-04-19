import json
from collections import Counter
import os

# Path to the JSON file
SAVE_PATH = os.path.join('.', 'nuscenes_blobs', 'train_samples.json')

def count_labels(json_path):
    """
    Reads the JSON dataset file and counts occurrences of each label in gt_labels.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    label_counter = Counter()
    
    for sample in data:
        labels = sample.get("gt_labels", [])
        label_counter.update(labels)
    
    return label_counter

if __name__ == '__main__':
    # Check if the file exists
    if not os.path.exists(SAVE_PATH):
        print(f"File not found: {SAVE_PATH}")
    else:
        counts = count_labels(SAVE_PATH)
        print("Counts for each label:")
        for label, count in counts.items():
            print(f"{label}: {count}")
