import numpy as np
from src.segmentation.segmenter import Segmenter
import os


#labels = np.load("data/processed/clusters/labels_frames.npy")
#timestamps = np.load("data/processed/clusters/timestamps_frames.npy")

labels = np.load("data/processed/clusters/labels_pixels.npy")
timestamps = np.load("data/processed/clusters/timestamps_pixels.npy")


print(f"Загрузили {len(labels)} меток и {len(timestamps)} временных точек")

segmenter = Segmenter(time_threshold=0.01)
segments = segmenter.segment(timestamps, labels)

print(f"\n Найдено {len(segments)} сегментов:\n")
for i, (start, end) in enumerate(segments):
    t_start = timestamps[start] * 86400
    t_end = timestamps[end] * 86400
    print(f" Сегмент {i + 1}: пиксели {start} → {end}  | время: {t_start:.1f}s → {t_end:.1f}s")
