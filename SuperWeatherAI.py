import requests 
import pandas as pd 
import numpy as np
from tkinter import * 

from keras.models import Sequential
from keras.layers import Dense, Input

x_train = np.array([
    [20.0, 50.0, 0.0],
    [25.0, 40.0, 0.0],
    [15.0, 80.0, 5.0],
    [10.0, 90.0, 12.0],
    [30.0, 30.0, 0.0]
])

y_train = np.array([[22.0], [24.0], [13.0], [9.0], [31.0]])

model = Sequential([
    Input(shape=(3,)),
    Dense(units=8, activation='relu'),
    Dense(units=4, activation='relu'),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mse')

print("Инициализация ИИ: обучаем нейросеть...")
model.fit(x_train, y_train, epochs=500, verbose=0)
print("Нейросеть готова к прогнозам!\n")

def get_weather():
    raw_city = entry1.get().strip()

    if raw_city == "":
        result_label.config(text="Ошибка: Вы не ввели город!", fg='red')
        return 
    user_city = raw_city.replace(", ", ",").replace(" ", "+")

    result_label.config(text="Скачиваю погоду и считаю прогноз...", fg='black')
    root.update()

    url = f"https://wttr.in/{user_city}?format=j1"

    try: 
        response = requests.get(url, timeout=5)

        if response.status_code == 200: 
            data = response.json() 

            temp = float(data['current_condition'][0]['temp_C'])
            humid = int(data['current_condition'][0]['humidity'])
            precip = float(data['current_condition'][0]['precipMM'])

            live_data = np.array([[temp, humid, precip]])

            ai_prediction = model.predict(live_data, verbose=0)
            tomorrow_temp = ai_prediction[0][0] 

            temp_f = temp * 1.8 + 32 

            display_text = (
                f"Город: {user_city.capitalize()}\n"
                f"---Сегодня (Живые данные)---\n"
                f"Температура: {temp}°C ({temp_f:.1f}°F)\n"
                f"Влажность: {humid}%\n"
                f"Осадки: {precip} мм\n\n"
                f"---Завтра (Прогноз ИИ)---\n"
                f"Ожидается: {tomorrow_temp:.1f}°C"
            )

            result_label.config(text=display_text, fg='green')

        else:
            result_label.config(text=f"Город '{user_city}' не найден!", fg='red')

    except Exception as e: 
        print(f"Техническая ошибка для отладки: {e}")
        result_label.config(text="Ошибка сети! Проверьте соединение с интернетом", fg='red')

root = Tk()
root.title("ИИ Прогноз Погоды")
root.geometry("400x350")

title_label = Label(root, text="Введите город и, если нужно, страну на английском через запятую\n(например: London, Canada или Tokyo):", font=("Arial", 10), justify=CENTER)
title_label.pack(pady=10)

entry1 = Entry(root, width=20, font=("Arial", 12))
entry1.pack(pady=5)

btn = Button(root, text="Получить прогноз ИИ", command=get_weather, font=("Arial", 12, "bold"))
btn.pack(pady=10)

result_label = Label(root, text="Введите город и нажмите кнопку", font=("Arial", 11), justify=LEFT)
result_label.pack(pady=15)

root.mainloop()
