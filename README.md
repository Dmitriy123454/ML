Извлечение информативных кадров из видео
python scripts/run_video_processing.py
Извлекает и сохраняет только информативные кадры из исходного видео в папку data/processed/.

python scripts/run_feature_extraction.py
извлекает признаки

Оценка числа кластеров
python scripts/analyze_optimal_clusters.py

Кластеризация пикселей по разным методам
python scripts/run_pixel_clustering.py
Для каждого кадра выполняется кластеризация всех пикселей (яркость, координаты, время) различными алгоритмами (KMeans, GMM, HDBSCAN и др.).
Метки кластеров сохраняются в data/processed/clusters/labels_pixels_<method>.npy.

Сегментация 
python scripts/run_auto_segmentation.py

 Jupyter Notebook
 jupyter notebook
