import numpy as np
from abc import ABC, abstractmethod

class BaseTemporalCluster(ABC):
    def __init__(self, time_weight=0.2):
        self.time_weight = time_weight

    def _combine_features(self, X_vis, X_time):
        return np.hstack([
            X_vis * (1 - self.time_weight),
            X_time * self.time_weight
        ])

    @abstractmethod
    def cluster(self, X_vis, X_time):
        pass
