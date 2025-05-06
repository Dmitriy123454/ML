from sklearn.cluster import MiniBatchKMeans
from .base_clustering import BaseTemporalCluster

class TemporalMiniBatchKMeans(BaseTemporalCluster):
    def __init__(self, n_clusters=5, batch_size=4096, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.batch_size = batch_size

    def cluster(self, X_vis, X_time):
        X = self._combine_features(X_vis, X_time)
        model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size, random_state=42)
        return model.fit_predict(X)
