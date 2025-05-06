import numpy as np

class Segmenter:
    def __init__(self, time_threshold=2.0):
        self.time_threshold = time_threshold

    def segment(self, timestamps, labels):
        segments = []
        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[start_idx] or (timestamps[i] - timestamps[start_idx]) > self.time_threshold:
                segments.append((start_idx, i-1))
                start_idx = i
        segments.append((start_idx, len(labels)-1))
        return segments
