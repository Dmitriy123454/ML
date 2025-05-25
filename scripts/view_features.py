import numpy as np
import pandas as pd

features = np.load("data/processed/features.npy")

print(f"Размер массива признаков: {features.shape}")

df = pd.DataFrame(features)

print("\nПризнаки для всех кадров:")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
