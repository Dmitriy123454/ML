from sklearn.cluster import KMeans
from .base_clustering import BaseTemporalCluster

class TemporalKMeans(BaseTemporalCluster):
    def __init__(self, n_clusters=5, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters

    def cluster(self, X_vis, X_time):
        X = self._combine_features(X_vis, X_time)
        model = KMeans(n_clusters=self.n_clusters, random_state=42)
        return model.fit_predict(X)
