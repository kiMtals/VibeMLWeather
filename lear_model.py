import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

CITIES = {
    "Moscow": (55.7512, 37.6184),
    "Saint_Petersburg": (59.9386, 30.3141),
    "Novosibirsk": (55.0415, 82.9346),
    "Yekaterinburg": (56.8519, 60.6122),
    "Kazan": (55.7887, 49.1221),
    "Nizhny_Novgorod": (56.3287, 44.002),
    "Chelyabinsk": (55.154, 61.4291),
    "Krasnoyarsk": (56.0184, 92.8672),
    "Samara": (53.2001, 50.15),
    "Ufa": (54.7431, 55.9678)
}

FEATURE_COLS = [
    'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
    'surface_pressure_mean', 'cloud_cover_mean', 'wind_speed_10m_max'
]

TARGET_COLS = [
    'temperature_2m_max', 'precipitation_sum',
    'surface_pressure_mean', 'wind_speed_10m_max'
]

def create_windows(data, features, targets, window_size=7, forecast_size=3):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_size + 1):
        x_window = data[features].iloc[i : i + window_size].values.flatten()
        y_window = data[targets].iloc[i + window_size : i + window_size + forecast_size].values.flatten()
        X.append(x_window)
        y.append(y_window)
    return np.array(X), np.array(y)

for city_name in CITIES.keys():
    file_name = f"weather_dataset_{city_name}.csv"
    
    if os.path.exists(file_name):
        print(f"Обучение для: {city_name}")
        df = pd.read_csv(file_name)
        
        X, y = create_windows(df, FEATURE_COLS, TARGET_COLS)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        joblib.dump(model, f"weather_rf_model_{city_name}.pkl")
        print(f"Сохранено: weather_rf_model_{city_name}.pkl")
    else:
        print(f"Файл не найден: {file_name}")