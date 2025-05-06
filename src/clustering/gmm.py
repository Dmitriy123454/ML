from sklearn.mixture import GaussianMixture
from .base_clustering import BaseTemporalCluster

class TemporalGMM(BaseTemporalCluster):
    def __init__(self, n_clusters=5, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters

    def cluster(self, X_vis, X_time):
        X = self._combine_features(X_vis, X_time)
        model = GaussianMixture(n_components=self.n_clusters, covariance_type='full', random_state=42)
        return model.fit(X).predict(X)
