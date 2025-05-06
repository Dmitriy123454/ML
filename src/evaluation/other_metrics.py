from sklearn.metrics import davies_bouldin_score

class OtherMetrics:
    @staticmethod
    def davies_bouldin(X, labels):
        return davies_bouldin_score(X, labels)
