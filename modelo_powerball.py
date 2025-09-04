import numpy as np
import tensorflow as tf
import os

# ====== Cargar modelo entrenado ======
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_powerball.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# ====== Columnas usadas ======
NUM_COLS = [f"Numero{i}" for i in range(1, 8)] + ["Powerball"]

# Valores min/max aproximados de Powerball AU
COL_MIN = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
COL_MAX = np.array([35, 35, 35, 35, 35, 35, 35, 20], dtype=np.float32)

def minmax_scale(a, cmin, cmax):
    return (a - cmin) / (cmax - cmin + 1e-8)

def minmax_inverse(a_scaled, cmin, cmax):
    return a_scaled * (cmax - cmin + 1e-8) + cmin

def postprocess_prediction(vec_float):
    pred = np.rint(vec_float).astype(int)
    pred = np.clip(pred, COL_MIN, COL_MAX)

    main, pb = pred[:7], int(pred[7])
    seen = set()
    for i in range(len(main)):
        while main[i] in seen:
            main[i] = min(main[i] + 1, int(COL_MAX[i]))
        seen.add(main[i])

    return main.tolist(), pb

def predict_from_last_draw():
    # Para demo: generamos un input ficticio (7 números + PB aleatorios)
    fake_last_draw = np.array([[5, 12, 18, 22, 27, 31, 34, 7]], dtype=np.float32)

    arr_sc = minmax_scale(fake_last_draw, COL_MIN, COL_MAX)
    out_sc = model.predict(arr_sc, verbose=0)
    out_real = minmax_inverse(out_sc, COL_MIN, COL_MAX)[0]

    return postprocess_prediction(out_real)