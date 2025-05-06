import numpy as np
import pandas as pd

# Загружаем признаки
features = np.load("data/processed/features.npy")

# Выводим форму массива
print(f"✅ Размер массива признаков: {features.shape}")

# Преобразуем в таблицу для красивого вывода
df = pd.DataFrame(features)

# Выводим все признаки
print("\n🔍 Признаки для всех кадров:")

# Выводим весь датафрейм без ограничений
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
