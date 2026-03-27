import customtkinter as ctk
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CITIES = {
    "Москва": ("Moscow", 55.7512, 37.6184),
    "Санкт-Петербург": ("Saint_Petersburg", 59.9386, 30.3141),
    "Новосибирск": ("Novosibirsk", 55.0415, 82.9346),
    "Екатеринбург": ("Yekaterinburg", 56.8519, 60.6122),
    "Казань": ("Kazan", 55.7887, 49.1221),
    "Нижний Новгород": ("Nizhny_Novgorod", 56.3287, 44.002),
    "Челябинск": ("Chelyabinsk", 55.154, 61.4291),
    "Красноярск": ("Krasnoyarsk", 56.0184, 92.8672),
    "Самара": ("Samara", 53.2001, 50.15),
    "Уфа": ("Ufa", 54.7431, 55.9678)
}

class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ML Прогноз Погоды")
        self.geometry("600x650")
        
        self.title_label = ctk.CTkLabel(
            self, 
            text="ИИ Прогноз Погоды", 
            font=("Arial", 26, "bold")
        )
        self.title_label.pack(pady=20)
        
        self.city_var = ctk.StringVar(value="Москва")
        self.city_dropdown = ctk.CTkOptionMenu(
            self,
            values=list(CITIES.keys()),
            variable=self.city_var,
            font=("Arial", 14),
            width=200
        )
        self.city_dropdown.pack(pady=5)

        self.predict_btn = ctk.CTkButton(
            self, 
            text="Получить детальный прогноз", 
            font=("Arial", 16, "bold"),
            height=45,
            command=self.make_prediction
        )
        self.predict_btn.pack(pady=10)
        
        self.result_box = ctk.CTkTextbox(
            self, 
            width=550, 
            height=400, 
            font=("Consolas", 16), 
            state="disabled",
            wrap="none"
        )
        self.result_box.pack(pady=20, padx=20)

        self.result_box.tag_config("time", foreground="#FFA500")
        self.result_box.tag_config("success", foreground="#32CD32")
        self.result_box.tag_config("error", foreground="#FF4500")
        self.result_box.tag_config("header", foreground="#ADD8E6")

    def print_to_box(self, text, tag=None):
        self.result_box.configure(state="normal")
        if tag:
            self.result_box.insert("end", text + "\n", tag)
        else:
            self.result_box.insert("end", text + "\n")
        self.result_box.configure(state="disabled")
        self.update()

    def make_prediction(self):
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.configure(state="disabled")
        
        selected_city_rus = self.city_var.get()
        city_eng, lat, lon = CITIES[selected_city_rus]
        
        self.print_to_box(f"Город: {selected_city_rus}", "header")
        self.print_to_box("Подключение к метеостанции...", "time")
        
        try:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=6)
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily": [
                    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                    "surface_pressure_mean", "cloud_cover_mean", "wind_speed_10m_max"
                ],
                "timezone": "auto"
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                self.print_to_box(f"Ошибка API: {response.status_code}", "error")
                return
            
            self.print_to_box("Данные получены.", "success")
            self.print_to_box("Запуск нейросети...", "time")
            
            df_recent = pd.DataFrame(response.json()["daily"])
            features = [
                'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
                'surface_pressure_mean', 'cloud_cover_mean', 'wind_speed_10m_max'
            ]
            X_input = df_recent[features].values.flatten().reshape(1, -1)
            
            model_filename = f"weather_rf_model_{city_eng}.pkl"
            model = joblib.load(model_filename)
            prediction = model.predict(X_input)
            
            forecast_matrix = prediction[0].reshape(3, 4)
            
            self.print_to_box("\n" + "="*48, "header")
            self.print_to_box("   ПРОГНОЗ НА СЛЕДУЮЩИЕ 3 ДНЯ (ИИ)   ", "header")
            self.print_to_box("="*48 + "\n", "header")
            
            for i, day_data in enumerate(forecast_matrix, start=1):
                temp, precip, pressure, wind = day_data
                forecast_date = (end_date + timedelta(days=i)).strftime("%d.%m.%Y")
                
                self.print_to_box(f"{forecast_date} (День {i}):", "time")
                self.print_to_box(f"  Температура: {temp:>6.1f} °C")
                self.print_to_box(f"  Осадки:      {precip:>6.1f} мм")
                self.print_to_box(f"  Давление:    {pressure:>6.1f} гПа")
                self.print_to_box(f"  Ветер (макс):{wind:>6.1f} км/ч\n")
                
            self.print_to_box("Прогноз успешно сформирован!", "success")
            
        except FileNotFoundError:
            self.print_to_box(f"\nОшибка: Модель '{model_filename}' не найдена.", "error")
        except Exception as e:
            self.print_to_box(f"\nКритическая ошибка: {str(e)}", "error")

app = WeatherApp()
app.mainloop()