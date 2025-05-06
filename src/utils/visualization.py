import matplotlib.pyplot as plt

class ClusterVisualizer:
    @staticmethod
    def plot_temporal_clusters(timestamps, labels):
        plt.figure(figsize=(12, 6))
        plt.scatter(timestamps, [1]*len(timestamps), c=labels, cmap='tab10')
        plt.xlabel('Time (seconds)')
        plt.yticks([])
        plt.title('Temporal Cluster Distribution')
        plt.show()
