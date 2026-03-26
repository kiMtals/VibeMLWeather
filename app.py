import customtkinter as ctk
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Настройки внешнего вида (темная тема)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Настройка главного окна
        self.title("ML Прогноз Погоды")
        self.geometry("550x550")
        self.resizable(False, False)
        
        # Заголовок
        self.title_label = ctk.CTkLabel(
            self, 
            text="Предсказание погоды на базе ИИ", 
            font=("Arial", 22, "bold")
        )
        self.title_label.pack(pady=(20, 10))
        
        # Кнопка запуска
        self.predict_btn = ctk.CTkButton(
            self, 
            text="Получить прогноз на 3 дня", 
            font=("Arial", 14, "bold"),
            height=40,
            command=self.make_prediction
        )
        self.predict_btn.pack(pady=10)
        
        # Текстовое поле для вывода результата
        self.result_box = ctk.CTkTextbox(
            self, 
            width=480, 
            height=350, 
            font=("Consolas", 15),
            state="disabled" # Делаем поле только для чтения
        )
        self.result_box.pack(pady=10)

    def print_to_box(self, text):
        """Вспомогательная функция для печати текста в графическое окно"""
        self.result_box.configure(state="normal")
        self.result_box.insert("end", text + "\n")
        self.result_box.configure(state="disabled")
        self.update() # Обновляем интерфейс в реальном времени

    def make_prediction(self):
        # Очищаем поле перед новым прогнозом
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.configure(state="disabled")
        
        self.print_to_box("⏳ Подключение к метеостанции и загрузка данных...")
        
        try:
            # 1. Запрашиваем данные за 7 дней
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=6)
            
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            lat, lon = 55.7512, 37.6184 # Координаты (Москва)
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": start_str, "end_date": end_str,
                "daily": [
                    "temperature_2m_max", 
                    "temperature_2m_min",     # <-- добавили
                    "precipitation_sum",
                    "surface_pressure_mean", 
                    "cloud_cover_mean",       # <-- добавили
                    "wind_speed_10m_max"
                ],
                "timezone": "auto"
            }
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                self.print_to_box(f"❌ Ошибка API: {response.status_code}")
                return
                
            data = response.json()
            df_recent = pd.DataFrame(data["daily"])
            
            if len(df_recent) != 7:
                self.print_to_box("❌ Ошибка: получено неверное количество дней.")
                return
                
            self.print_to_box("✅ Данные получены. Запуск нейросети...")
            
            # 2. Подготовка данных
            features = [
                'temperature_2m_max', 
                'temperature_2m_min',     # <-- добавили
                'precipitation_sum',
                'surface_pressure_mean', 
                'cloud_cover_mean',       # <-- добавили
                'wind_speed_10m_max'
            ]
            X_input = df_recent[features].values.flatten().reshape(1, -1)
            
            # 3. Загрузка модели и прогноз
            model = joblib.load("weather_rf_model3.pkl")
            prediction = model.predict(X_input)
            
            forecast_matrix = prediction[0].reshape(3, 4)
            
            self.print_to_box("\n" + "═" * 40)
            self.print_to_box("🌤 ПРОГНОЗ НА СЛЕДУЮЩИЕ 3 ДНЯ:")
            self.print_to_box("═" * 40 + "\n")
            
            # 4. Вывод результата
            for i, day_data in enumerate(forecast_matrix, start=1):
                temp, precip, pressure, wind = day_data
                forecast_date = (end_date + timedelta(days=i)).strftime("%d.%m.%Y")
                
                self.print_to_box(f"📅 День {i} ({forecast_date}):")
                self.print_to_box(f"  🌡 Температура: {temp:>5.1f} °C")
                self.print_to_box(f"  🌧 Осадки:      {precip:>5.1f} мм")
                self.print_to_box(f"  🧭 Давление:    {pressure:>5.1f} гПа")
                self.print_to_box(f"  💨 Ветер (макс):{wind:>5.1f} км/ч\n")
                
            self.print_to_box("Готово!")
            
        except FileNotFoundError:
            self.print_to_box("\n❌ Ошибка: Файл модели 'weather_rf_model.pkl' не найден.")
            self.print_to_box("Сначала обучите модель!")
        except Exception as e:
            self.print_to_box(f"\n❌ Произошла ошибка: {str(e)}")

# Запуск приложения
if __name__ == "__main__":
    app = WeatherApp()
    app.mainloop()