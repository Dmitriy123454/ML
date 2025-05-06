from sklearn.cluster import AgglomerativeClustering
from .base_clustering import BaseTemporalCluster

class TemporalHierarchical(BaseTemporalCluster):
    def __init__(self, n_clusters=5, linkage="average", affinity="l2", **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity  # 👈 обязательно добавляем

    def cluster(self, X_vis, X_time):
        X = self._combine_features(X_vis, X_time)
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=self.affinity
        )
        return model.fit_predict(X)
