from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np

class ClusterEvaluator:
    @staticmethod
    def evaluate(X, labels, time_indices):
        metrics = {
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'temporal_variance': np.var([np.mean(time_indices[labels == i]) for i in np.unique(labels)])
        }
        return metrics

    @staticmethod
    def silhouette(X, labels):
        return silhouette_score(X, labels)

    @staticmethod
    def calinski_harabasz(X, labels):
        return calinski_harabasz_score(X, labels)
