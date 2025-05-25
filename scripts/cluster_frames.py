import numpy as np
import json
import os
from sklearn.cluster import KMeans

from src.utils.feature_extraction import TemporalFeatureEngineer

features_path = "data/processed/features.npy"
metadata_path = "data/processed/metadata.json"
output_dir = "data/processed/clusters"
os.makedirs(output_dir, exist_ok=True)

features = np.load(features_path)

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(features)

np.save(os.path.join(output_dir, "labels_frames.npy"), labels)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

timestamps = [frame["timestamp"] for frame in metadata]
np.save(os.path.join(output_dir, "timestamps_frames.npy"), np.array(timestamps))

print("Кадровые метки и времена сохранены!")
