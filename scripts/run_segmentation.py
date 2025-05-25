import os
import cv2
import numpy as np
import shutil
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

input_dir = "data/processed"

output_base_dir = "outputs/segmentation"
methods = ["kmeans", "hierarchical", "dbscan"]


n_clusters = 5
eps_dbscan = 5
min_samples_dbscan = 10


brightness_weight = 1.0
coordinate_weight = 0.1


full_width = 1280
full_height = 966


small_width = 320
small_height = 241

def clear_output_dir():
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

def extract_pixel_features(image):
    h, w = image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    coordinates = np.stack([X.ravel(), Y.ravel()], axis=1) * coordinate_weight
    brightness = image.ravel().reshape(-1, 1) * brightness_weight
    features = np.hstack([brightness, coordinates])
    return features

def cluster_pixels(features, method):
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "dbscan":
        model = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    labels = model.fit_predict(features)
    return labels

def get_contrast_color(idx):
    contrast_colors = [
        (255, 0, 0),
        (0, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (255, 165, 0),
    ]
    return contrast_colors[idx % len(contrast_colors)]

def visualize_colored_boundaries(original_image, labels, shape):
    segmented = labels.reshape(shape)
    unique_labels = np.unique(labels)
    overlay = original_image.copy()

    for idx, label in enumerate(unique_labels):
        if label == -1:
            continue
        mask = (segmented == label)
        boundary = find_boundaries(mask, mode='thick')
        color = get_contrast_color(idx)

        coords = np.argwhere(boundary)
        for y, x in coords:
            if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
                overlay[y, x] = color

    return overlay

def process_all_frames():
    frame_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])

    for method in methods:
        output_dir = os.path.join(output_base_dir, method)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n Кластеризация методом {method}")

        for frame_file in frame_files:
            frame_path = os.path.join(input_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f" Проблема с загрузкой файла: {frame_path}")
                continue

            h, w = frame.shape[:2]
            if (w, h) != (full_width, full_height):
                frame_resized = cv2.resize(frame, (full_width, full_height))
            else:
                frame_resized = frame.copy()

            if method == "hierarchical":
                frame_to_process = cv2.resize(frame_resized, (small_width, small_height))
                print(f" Hierarchical: уменьшили до {small_width}x{small_height}")
            else:
                frame_to_process = frame_resized

            gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)

            features = extract_pixel_features(gray)
            labels = cluster_pixels(features, method)

            overlay = visualize_colored_boundaries(frame_resized, labels, gray.shape)

            output_path = os.path.join(output_dir, f"segmented_{frame_file}")
            cv2.imwrite(output_path, overlay)
            print(f"Сохранено: {output_path}")

if __name__ == "__main__":
   # print("Очистка старых результатов...")
   #  clear_output_dir()
    process_all_frames()
    print("\nОбработка завершена!")
