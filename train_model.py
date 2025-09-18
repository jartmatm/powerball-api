import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from url_years import urls
from tensorflow.keras import layers, models

# ========= 1) Scraping =========
all_data = []
for year, url in urls.items():
    print(f"Procesando {year}...")
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        ball = soup.find_all('li', class_='ball ball -b280')
        powerball = soup.find_all('li', class_='ball powerball -b280')

        for i in range(0, len(ball), 7):
            numeros = [int(q.text) for q in ball[i:i+7]]
            pb = int(powerball[i // 7].text) if (i // 7) < len(powerball) else 0
            all_data.append(numeros + [pb])

df = pd.DataFrame(all_data, columns=[f"Numero{i}" for i in range(1, 8)] + ["Powerball"])

# ========= 2) Preparación =========
X_rows = df.iloc[::2].reset_index(drop=True)
Y_rows = df.iloc[1::2].reset_index(drop=True)
m = min(len(X_rows), len(Y_rows))
X, Y = X_rows.iloc[:m].values, Y_rows.iloc[:m].values

X = X.astype(np.float32)
Y = Y.astype(np.float32)

both = np.vstack([X, Y])
col_min, col_max = both.min(axis=0), both.max(axis=0)

def minmax_scale(a, cmin, cmax):
    return (a - cmin) / (cmax - cmin + 1e-8)

X_sc = minmax_scale(X, col_min, col_max)
Y_sc = minmax_scale(Y, col_min, col_max)

# ========= 3) Modelo =========
model = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(500, activation="relu"),
    layers.Dense(250, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(8, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse")
print("Entrenando modelo...")
model.fit(X_sc, Y_sc, epochs=200, batch_size=256, verbose=1)

# ========= 4) Guardar =========
model.save("modelo_powerball.keras")
df.to_csv("historical_draws.csv", index=False)
print("Modelo guardado como modelo_powerball.keras")
