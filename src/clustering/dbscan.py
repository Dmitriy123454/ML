from sklearn.cluster import DBSCAN
from src.clustering.base_clustering import BaseTemporalCluster
import numpy as np

class TemporalDBSCAN(BaseTemporalCluster):
    def __init__(self, eps=12, min_samples=2, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, X_vis, X_time):
        X = self._combine_features(X_vis, X_time)
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.model.fit_predict(X)


        unique, counts = np.unique(labels, return_counts=True)
        print(f"[DBSCAN] Распределение меток: {dict(zip(unique, counts))}")

        return labels
