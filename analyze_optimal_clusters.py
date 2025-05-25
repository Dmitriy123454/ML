import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.evaluation.elbow_method import ElbowMethod
from src.evaluation.cluster_evaluation import ClusterEvaluator
from src.evaluation.other_metrics import OtherMetrics

features_path = "../PythonProjec/PythonProject9/data/processed/features.npy"
output_dir = "outputs/analysis"
os.makedirs(output_dir, exist_ok=True)
elbow_plot_path = os.path.join(output_dir, "elbow_plot.png")
combined_plot_path = os.path.join(output_dir, "combined_plot.png")
metrics_txt_path = os.path.join(output_dir, "cluster_metrics.txt")

if __name__ == "__main__":
    print(" Загрузка признаков...")
    X = np.load(features_path)
    print(f" Признаков: {X.shape}")

    min_k = 2
    max_k = 10

    wcss = []
    silhouette_list = []
    calinski_list = []
    davies_list = []

    print(" Расчёт метрик...")
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)

        wcss.append(model.inertia_)
        silhouette_list.append(ClusterEvaluator.silhouette(X, labels))
        calinski_list.append(ClusterEvaluator.calinski_harabasz(X, labels))
        davies_list.append(OtherMetrics.davies_bouldin(X, labels))

    ElbowMethod.plot_elbow(wcss, save_path=elbow_plot_path)
    print(f"Сохранён elbow plot → {elbow_plot_path}")

    with open(metrics_txt_path, "w") as f:
        f.write("Clusters\tSilhouette\tCalinski-Harabasz\tDavies-Bouldin\n")
        for k, s, c, d in zip(range(min_k, max_k + 1), silhouette_list, calinski_list, davies_list):
            f.write(f"{k}\t{s:.4f}\t{c:.2f}\t{d:.4f}\n")
    print(f"Метрики сохранены → {metrics_txt_path}")

    ElbowMethod.plot_combined(
        wcss=wcss,
        silhouette_scores=silhouette_list,
        calinski_scores=calinski_list,
        save_path=combined_plot_path,
        cluster_range=range(min_k, max_k + 1)
    )
    print(f"Сводный график сохранён → {combined_plot_path}")

    print("\nАнализ завершён!")
