from sklearn.cluster import SpectralClustering
from .base_clustering import BaseTemporalCluster

class TemporalSpectral(BaseTemporalCluster):
    def __init__(self, n_clusters=5, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters

    def cluster(self, X_vis, X_time):
        X = self._combine_features(X_vis, X_time)
        model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=50,
            assign_labels='kmeans',
            random_state=42
        )
        return model.fit_predict(X)



