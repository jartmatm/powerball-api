import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import os
from bs4 import BeautifulSoup
from url_years import urls
from tensorflow.keras import layers, models

all_data = []  # Lista para guardar todos los registros

for year, url in urls.items():  # Iterar sobre cada año y su URL
    print(f'procesando {year}...')
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        ball = soup.find_all('li', class_='ball ball -b280')
        powerball = soup.find_all('li', class_='ball powerball -b280')
        draw = soup.find_all('td', class_='noBefore colour date-row')

        for i in range(0, len(ball), 7):
            bloque_numeros = ball[i:i+7]
            numeros = [q.text for q in bloque_numeros]

            PW = powerball[i // 7].text if (i // 7) < len(powerball) else "Powerball Desconocida"
            DR = draw[i // 7].text.strip() if (i // 7) < len(draw) else "Draw is missing"

            registro = {
                'Sorteo': DR,
                'Powerball': PW,
                'Year': year
            }

            for idx, num in enumerate(numeros, start=1):
                registro[f'Numero{idx}'] = num

            all_data.append(registro)

    else:
        print(f'Error al acceder a la pagina {url}')

df = pd.DataFrame(all_data)

cols = df.columns.tolist()
cols.remove('Powerball')
cols.append('Powerball')
df = df[cols]

df['Numero de Sorteo'] = df['Sorteo'].str.extract(r'([\d,]+)')
df['Fecha'] = df['Sorteo'].str.extract(r'\d+(\w+\s\d{1,2}\s\w+\s\d{4})')
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df = df.drop(columns=['Sorteo'])

columnas_numeros = ["Numero1", "Numero2", "Numero3", "Numero4", "Numero5", "Numero6", "Numero7", "Powerball"]

df = df.sort_values("Fecha")
df = df.reset_index(drop=True)
for col in columnas_numeros:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[columnas_numeros] = df[columnas_numeros].fillna(0)
df[columnas_numeros] = df[columnas_numeros].astype(int)

# ======================= Modelo con TensorFlow/Keras =======================



# ======== 1) Preparar datos ========
num_cols = [f"Numero{i}" for i in range(1, 8)] + ["Powerball"]
df_nums = df[num_cols].copy().reset_index(drop=True)

# Convertir a enteros
for c in num_cols:
    df_nums[c] = pd.to_numeric(df_nums[c], errors="coerce")
df_nums = df_nums.dropna(subset=num_cols).astype(int).reset_index(drop=True)

# Pares alternos
X_rows = df_nums.iloc[::2].copy()
Y_rows = df_nums.iloc[1::2].copy()
m = min(len(X_rows), len(Y_rows))
X_rows, Y_rows = X_rows.iloc[:m].reset_index(drop=True), Y_rows.iloc[:m].reset_index(drop=True)

X_np, Y_np = X_rows.values.astype(np.float32), Y_rows.values.astype(np.float32)

# Escalado MinMax
both = np.vstack([X_np, Y_np])
col_max, col_min = both.max(axis=0).clip(min=1.0), both.min(axis=0)

def minmax_scale(a, cmin, cmax):
    return (a - cmin) / (cmax - cmin + 1e-8)

def minmax_inverse(a_scaled, cmin, cmax):
    return a_scaled * (cmax - cmin + 1e-8) + cmin

X_sc, Y_sc = minmax_scale(X_np, col_min, col_max), minmax_scale(Y_np, col_min, col_max)

# ======== 2) Definir modelo en TensorFlow ========
model = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(500, activation="relu"),
    layers.Dense(250, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(8, activation="sigmoid")  # salida 0-1
])

model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])


# ======== 3) Entrenamiento ========
print("Entrenando el modelo...")
history = model.fit(X_sc, Y_sc, epochs=1500, batch_size=512, shuffle=False, verbose=0)

# ======== 4) Guardar el modelo entrenado ========

print("Directorio actual:", os.getcwd())
model.save("modelo_powerball.keras")
print("Modelo guardado como modelo_powerball.keras")

# ======== 5) Helpers ========
def postprocess_prediction(vec_float):
    real_col_max = df_nums[num_cols].max().values
    real_col_min = df_nums[num_cols].min().values

    pred = np.rint(vec_float).astype(int)

    for i in range(8):
        lo, hi = int(real_col_min[i]), int(real_col_max[i])
        pred[i] = np.clip(pred[i], lo, hi)

    main, pb = pred[:7], int(pred[7])

    # Asegurar unicidad en los 7 principales
    seen = set()
    for i in range(len(main)):
        while main[i] in seen:
            main[i] = min(main[i] + 1, int(real_col_max[i]))
        seen.add(main[i])

    return main.tolist(), pb

def predict_from_input(nums7_plus_pb):
    arr = np.array(nums7_plus_pb, dtype=np.float32).reshape(1, -1)
    arr_sc = minmax_scale(arr, col_min, col_max)
    out_sc = model.predict(arr_sc, verbose=0)
    out_real = minmax_inverse(out_sc, col_min, col_max)[0]
    return postprocess_prediction(out_real)

def predict_from_last_draw():
    last = df_nums.iloc[-1][num_cols].values.astype(np.float32).reshape(1, -1)
    last_sc = minmax_scale(last, col_min, col_max)
    out_sc = model.predict(last_sc, verbose=0)
    out_real = minmax_inverse(out_sc, col_min, col_max)[0]
    return postprocess_prediction(out_real)

# ======== 6) Ejemplo ========
nums_pred2, pb_pred2 = predict_from_last_draw()
print("Versión B → Próximo sugerido desde el último sorteo:", nums_pred2, "| PB:", pb_pred2)
