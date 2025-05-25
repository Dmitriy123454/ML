import numpy as np
import os
import cv2
import json
import shutil
from skimage.measure import label as skimage_label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt

METHODS_TO_SEGMENT = [
    "kmeans",
    "minibatch_kmeans",
    "gmm",
    "hdbscan",
   # "hierarchical",
    #"spectral"
]

FRAME_SIZE = (640, 480)
CLUSTER_DIR = os.path.join("data", "processed", "clusters")
OUT_BASE = os.path.join("outputs", "pixel_rule_segments")
FRAME_DIR = "data/processed"
MIN_SEGMENT_SIZE = 200
def segment_by_cluster_label_and_connectivity(labels_1d, w, h, min_size=200):

    seg_mask = np.zeros((h, w), dtype=int)
    labels_2d = labels_1d.reshape(h, w)
    seg_id = 1
    for cluster_id in np.unique(labels_2d):
        if cluster_id == -1:
            continue
        mask = (labels_2d == cluster_id)
        labeled = skimage_label(mask, connectivity=1)
        for region in regionprops(labeled):
            if region.area >= min_size:
                for coords in region.coords:
                    seg_mask[coords[0], coords[1]] = seg_id
                seg_id += 1
    return seg_mask

def visualize_segments(seg_mask, orig_frame, out_path):
    if orig_frame.shape[:2] != seg_mask.shape:
        orig_frame = cv2.resize(orig_frame, (seg_mask.shape[1], seg_mask.shape[0]))
    image = label2rgb(seg_mask, image=orig_frame, bg_label=0, alpha=0.3)
    plt.figure(figsize=(7, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Сегментация по кластерам и связности")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    for method in METHODS_TO_SEGMENT:
        print(f"\n=== Сегментация по меткам кластеров для метода: {method} ===")
        labels_path = os.path.join(CLUSTER_DIR, f"labels_pixels_{method}.npy")
        if not os.path.exists(labels_path):
            print(f" Нет файла {labels_path}")
            continue

        labels = np.load(labels_path)
        w, h = FRAME_SIZE
        pixels_per_frame = w * h
        n_frames = len(labels) // pixels_per_frame

        vis_dir = os.path.join(OUT_BASE, method)
        if os.path.exists(vis_dir):
            shutil.rmtree(vis_dir)
        os.makedirs(vis_dir, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith('.jpg')])

        for fidx in range(n_frames):
            start = fidx * pixels_per_frame
            end = (fidx + 1) * pixels_per_frame
            frame_labels = labels[start:end]

            if fidx < len(frame_files):
                img_path = os.path.join(FRAME_DIR, frame_files[fidx])
                frame = cv2.imread(img_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = np.zeros((h, w, 3), dtype=np.uint8)

            seg_mask = segment_by_cluster_label_and_connectivity(
                frame_labels, w, h, min_size=MIN_SEGMENT_SIZE
            )

            out_img = os.path.join(vis_dir, f"frame_{fidx:03d}_rule_segments.png")
            visualize_segments(seg_mask, frame, out_img)
            print(f" Сегментация кадра {fidx+1} по правилу сохранена: {out_img}")

if __name__ == "__main__":
    main()
