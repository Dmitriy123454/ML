import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import numpy as np
import cv2
import time
from skimage.segmentation import find_boundaries

from src.clustering.kmeans import TemporalKMeans
from src.clustering.hierarchical import TemporalHierarchical
from src.clustering.hdbscan import TemporalHDBSCAN
from src.clustering.minibatch_kmeans import TemporalMiniBatchKMeans
from src.clustering.gmm import TemporalGMM
from src.clustering.spectral import TemporalSpectral

input_dir = "data/processed"
output_base_dir = "outputs/segmentation"
cluster_save_dir = "data/processed/clusters"
os.makedirs(cluster_save_dir, exist_ok=True)

full_width = 640
full_height = 480

n_clusters = 5
min_cluster_size_hdbscan = 500

brightness_weight = 1.0
coordinate_weight = 0.05
time_weight = 0.2

available_methods = [
    "kmeans",
    "minibatch_kmeans",
    "gmm",
    "hdbscan",
    #"hierarchical",
    #"spectral"
]

def clear_output_dirs():
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

def extract_pixel_features(image, timestamp_norm):
    h, w = image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32) * coordinate_weight
    brightness = image.ravel().reshape(-1, 1).astype(np.float32) * brightness_weight
    time_feature = np.full((h * w, 1), timestamp_norm * time_weight, dtype=np.float32)
    all_features = np.hstack([brightness, coords, time_feature])
    X_vis = np.hstack([brightness, coords])
    X_time = time_feature
    return X_vis, X_time, all_features

def get_contrast_color(idx):
    colors = [
        (255, 0, 0), (0, 0, 0), (255, 0, 255),
        (255, 255, 0), (255, 165, 0), (0, 255, 255),
        (128, 0, 128), (0, 255, 0)
    ]
    return colors[idx % len(colors)]

def visualize_boundaries(image, labels, shape):
    segmented = labels.reshape(shape)
    unique_labels = np.unique(labels)
    overlay = image.copy()
    for idx, label in enumerate(unique_labels):
        if label == -1:
            continue
        mask = (segmented == label)
        boundary = find_boundaries(mask, mode='thick')
        coords = np.argwhere(boundary)
        color = get_contrast_color(idx)
        for y, x in coords:
            overlay[y, x] = color
    return overlay

def cluster_pixels(X_vis, X_time, all_features, method, frame=None):
    if method == "kmeans":
        return TemporalKMeans(n_clusters=n_clusters).cluster(X_vis, X_time)
    elif method == "hierarchical":
        return TemporalHierarchical(n_clusters=n_clusters).cluster(X_vis, X_time)
    elif method == "hdbscan":
        return TemporalHDBSCAN(min_cluster_size=min_cluster_size_hdbscan).cluster(X_vis, X_time)
    elif method == "minibatch_kmeans":
        return TemporalMiniBatchKMeans(n_clusters=n_clusters).cluster(X_vis, X_time)
    elif method == "gmm":
        return TemporalGMM(n_clusters=n_clusters).cluster(X_vis, X_time)
    elif method == "spectral":
        return TemporalSpectral(n_clusters=n_clusters).cluster(X_vis, X_time)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

def process_frames(methods=None):
    frame_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
    if not frame_files:
        print("Нет кадров для обработки.")
        return

    methods = methods or available_methods
    total_start = time.time()

    for method in methods:
        print(f"\nКластеризация методом: {method}")
        method_start = time.time()
        method_dir = os.path.join(output_base_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        method_labels = []
        method_timestamps = []

        for frame_file in frame_files:
            frame_start = time.time()
            frame_path = os.path.join(input_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Пропуск: {frame_file}")
                continue

            resized = cv2.resize(frame, (full_width, full_height))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            ts_str = frame_file.split('_')[2][:6]
            timestamp = int(ts_str[:2]) * 3600 + int(ts_str[2:4]) * 60 + int(ts_str[4:6])
            t_norm = timestamp / (24 * 3600)

            X_vis, X_time, all_features = extract_pixel_features(gray, t_norm)
            labels = cluster_pixels(X_vis, X_time, all_features, method, frame=resized)

            uniq, counts = np.unique(labels, return_counts=True)
            print(f" [{frame_file}] метки: {dict(zip(uniq, counts))}")

            if np.all(labels == -1):
                print(f" Метод {method} не нашёл ни одного кластера — все пиксели = шум")

            vis = visualize_boundaries(resized, labels, gray.shape)
            out_path = os.path.join(method_dir, f"segmented_{frame_file}")
            cv2.imwrite(out_path, vis)

            method_labels.extend(labels.tolist())
            method_timestamps.extend([t_norm] * len(labels))

            frame_time = round(time.time() - frame_start, 2)
            print(f" Время кадра: {frame_time} сек")

        # Сохраняем метки отдельно для каждого метода
        np.save(os.path.join(cluster_save_dir, f"labels_pixels_{method}.npy"), np.array(method_labels))
        np.save(os.path.join(cluster_save_dir, f"timestamps_pixels_{method}.npy"), np.array(method_timestamps))
        print(f"Метки и времена для метода {method} сохранены!")

        method_time = round(time.time() - method_start, 2)
        print(f"Метод {method} завершён за {method_time} сек")

    total_time = round(time.time() - total_start, 2)
    print(f"\n Все методы завершены за {total_time} сек")

if __name__ == "__main__":
    # print(" Очистка результатов...")
    # clear_output_dirs()   Раскомментируйте, если нужно почистить
    process_frames()
    print("\n Готово!")
