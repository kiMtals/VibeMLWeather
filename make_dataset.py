import requests
import pandas as pd
import time

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

def fetch_historical_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
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
        daily_data = data["daily"]
        df = pd.DataFrame(daily_data)
        df['time'] = pd.to_datetime(df['time'])
        return df
    else:
        print(f"Ошибка: {response.status_code}")
        return None

for city_name, coords in CITIES.items():
    lat, lon = coords
    print(f"Скачиваем данные для: {city_name}...")
    
    df_weather = fetch_historical_weather(
        lat=lat, 
        lon=lon, 
        start_date="2020-01-01", 
        end_date="2023-12-31"
    )
    
    if df_weather is not None:
        file_name = f"weather_dataset_{city_name}.csv"
        df_weather.to_csv(file_name, index=False, encoding="utf-8")
        print(f"Сохранено: {file_name}")
    
    time.sleep(2)