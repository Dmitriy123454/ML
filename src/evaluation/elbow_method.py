import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ElbowMethod:
    @staticmethod
    def calculate_wcss(X, max_clusters=10):
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        return wcss

    @staticmethod
    def plot_elbow(wcss, save_path=None):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
        plt.title('Elbow Method For Optimal Number of Clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_combined(wcss, silhouette_scores, calinski_scores, save_path=None, cluster_range=None):
        if cluster_range is None:
            cluster_range = range(2, len(wcss) + 2)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_title("Определение оптимального числа кластеров")
        ax1.plot(cluster_range, wcss, 'bo-', label='WCSS (Elbow)')
        ax1.set_xlabel('Количество кластеров')
        ax1.set_ylabel('WCSS', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(cluster_range, silhouette_scores, 'rs--', label='Silhouette')
        ax2.plot(cluster_range, calinski_scores, 'g^--', label='Calinski-Harabasz')
        ax2.set_ylabel('Индексы качества', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
