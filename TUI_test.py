import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# 1. Определяем даты: берем 7 дней, заканчивая вчерашним днем
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=6)

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

print(f"Запрашиваем данные с {start_str} по {end_str}...")

# Координаты (Москва)
lat, lon = 55.7512, 37.6184

# Используем основной API (он умеет отдавать недавнее прошлое)
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start_str,
    "end_date": end_str,
    "daily": [
        "temperature_2m_max", 
        "temperature_2m_min", 
        "precipitation_sum",
        "surface_pressure_mean",
        "cloud_cover_mean",
        "wind_speed_10m_max"
    ],
    "timezone": "auto"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    df_recent = pd.DataFrame(data["daily"])
    
    # Проверяем, что пришло ровно 7 дней
    if len(df_recent) != 7:
        print(f"Ошибка: получено {len(df_recent)} дней вместо 7!")
    else:
        # 2. Подготовка данных в точности как при обучении
        features = [
            'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
            'surface_pressure_mean', 'cloud_cover_mean', 'wind_speed_10m_max'
        ]
        
        # Вытаскиваем значения и "сплющиваем" их в один ряд (42 значения)
        X_input = df_recent[features].values.flatten()
        
        # ВАЖНО: scikit-learn ожидает таблицу (матрицу), где строки - это примеры.
        # У нас всего 1 пример (наши текущие 7 дней). 
        # Функция reshape(1, -1) превращает одномерный массив в таблицу из одной строки.
        X_input = X_input.reshape(1, -1)
        
        # 3. Загрузка модели и прогнозирование
        print("Загружаем модель...")
        model = joblib.load("weather_rf_model3.pkl")
        
        print("Делаем прогноз...\n")
        prediction = model.predict(X_input)
        
        # В prediction[0] сейчас лежат 3 числа (максимальная температура на 3 дня)
        TARGET_COLS = [
            'temperature_2m_max', 'precipitation_sum', 
            'surface_pressure_mean', 'wind_speed_10m_max'
        ]

        # Превращаем плоский массив из 12 чисел обратно в таблицу 3x4
        forecast_matrix = prediction[0].reshape(3, len(TARGET_COLS))
        
        print("-" * 45)
        print("🌤 ПОЛНЫЙ ПРОГНОЗ НА СЛЕДУЮЩИЕ 3 ДНЯ:")
        print("-" * 45)
        
        for i, day_data in enumerate(forecast_matrix, start=1):
            # Распаковываем данные конкретного дня (теперь их 4!)
            temp, precip, pressure, wind = day_data
            
            forecast_date = (end_date + timedelta(days=i)).strftime("%d.%m.%Y")
            print(f"День {i} ({forecast_date}):")
            print(f"  🌡 Температура: {temp:.1f} °C")
            print(f"  🌧 Осадки:      {precip:.1f} мм")
            print(f"  🧭 Давление:    {pressure:.1f} гПа")
            print(f"  💨 Ветер (макс):{wind:.1f} км/ч\n") # <-- выводим ветер
else:
    print(f"Ошибка API: {response.status_code}")