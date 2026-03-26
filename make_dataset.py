import requests
import pandas as pd

def fetch_historical_weather(lat, lon, start_date, end_date):
    # Обращаемся к архивному API Open-Meteo
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Параметры запроса (собираем максимальную и минимальную температуру, и осадки)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max", 
            "temperature_2m_min", 
            "precipitation_sum",
            "surface_pressure_mean", # <-- добавили
            "cloud_cover_mean",      # <-- добавили
            "wind_speed_10m_max"     # <-- добавили
        ],
        "timezone": "auto"
    }
    
    print("Скачиваем данные, подождите...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Берем только раздел daily (дневные показатели)
        daily_data = data["daily"]
        
        # Превращаем словарь в удобную таблицу pandas
        df = pd.DataFrame(daily_data)
        
        # Преобразуем колонку со временем в формат даты
        df['time'] = pd.to_datetime(df['time'])
        
        return df
    else:
        print(f"Ошибка: {response.status_code}")
        return None

# Координаты (например, Москва)
latitude = 55.7512
longitude = 37.6184

# Собираем данные за 4 года для обучения
df_weather = fetch_historical_weather(
    lat=latitude, 
    lon=longitude, 
    start_date="2020-01-01", 
    end_date="2023-12-31"
)

# Смотрим, что получилось
if df_weather is not None:
    print(df_weather.head())
    print(f"\nВсего собрано дней: {len(df_weather)}")
    
    # --- НОВЫЙ КОД ---
    # Сохраняем датафрейм в CSV файл
    file_name = "weather_dataset.csv"
    df_weather.to_csv(file_name, index=False, encoding="utf-8")
    
    print(f"Данные успешно сохранены в файл: {file_name}!")