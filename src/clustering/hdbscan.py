from src.clustering.base_clustering import BaseTemporalCluster
import numpy as np

try:
    import hdbscan
except ImportError:
    raise ImportError("Установи hdbscan: pip install hdbscan")

class TemporalHDBSCAN(BaseTemporalCluster):
    def __init__(self, min_cluster_size=500, **kwargs):
        super().__init__(**kwargs)
        self.min_cluster_size = min_cluster_size

    def cluster(self, X_vis, X_time):
        X = self._combine_features(X_vis, X_time)
        self.model = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        labels = self.model.fit_predict(X)

        unique, counts = np.unique(labels, return_counts=True)
        print(f"[HDBSCAN] Метки кластеров: {dict(zip(unique, counts))}")
        return labels
