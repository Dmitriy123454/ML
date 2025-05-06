from sklearn.metrics import silhouette_score

class SilhouetteMetric:
    @staticmethod
    def compute(X, labels):
        return silhouette_score(X, labels)
