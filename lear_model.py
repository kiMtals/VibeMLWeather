import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Загрузка данных
df = pd.read_csv("weather_dataset.csv")

# Задаем признаки и то, что будем предсказывать. 
# Для простоты и точности начнем с предсказания максимальной температуры.
FEATURE_COLS = [
    'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
    'surface_pressure_mean', 'cloud_cover_mean', 'wind_speed_10m_max'
]
TARGET_COLS = ['temperature_2m_max', 'precipitation_sum', 'surface_pressure_mean','wind_speed_10m_max']

def create_windows(data, features, targets, window_size=7, forecast_size=3):
    X, y = [], []
    
    for i in range(len(data) - window_size - forecast_size + 1):
        # Входные данные (X) остаются теми же: 7 дней * 6 признаков = 42 числа
        x_window = data[features].iloc[i : i + window_size].values.flatten()
        
        # Целевые данные (y): берем 3 дня и 3 нужные колонки
        # И ТОЖЕ сплющиваем! Получится 3 * 3 = 9 чисел
        y_window = data[targets].iloc[i + window_size : i + window_size + forecast_size].values.flatten()
        
        X.append(x_window)
        y.append(y_window)
        
    return np.array(X), np.array(y)


print("Размечаем данные...")
# Не забудьте обновить вызов функции:

X, y = create_windows(df, FEATURE_COLS, TARGET_COLS)

# 3. Разделение на обучение и тест
# ВАЖНО: shuffle=False! Для временных рядов нельзя перемешивать дни, 
# иначе модель "подсмотрит" в будущее при обучении.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Создание и обучение модели
# n_estimators - количество деревьев. 100 обычно достаточно.
model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Обучаем модель (это может занять пару секунд)...")
model.fit(X_train, y_train)

# 5. Оценка качества
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Обучение завершено!")
print(f"Средняя ошибка (MAE): {mae:.2f} °C") 
# Если MAE, например, 2.5, значит модель в среднем ошибается на 2.5 градуса

# 6. Сохранение модели
model_filename = "weather_rf_model3.pkl"
joblib.dump(model, model_filename)
print(f"Модель успешно сохранена в файл: {model_filename}")