# modelo_powerball.py
import os
import numpy as np
import tensorflow as tf

# Ruta del modelo en la misma carpeta (relativa)
MODEL_FILENAME = "modelo_powerball.keras"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# Rutas opcionales: un CSV de históricos (generado por train_model.py si lo usas)
HIST_CSV = os.path.join(os.path.dirname(__file__), "historical_draws.csv")

# Rango real Australia Powerball (ajusta si cambian)
COL_MIN = np.array([1,1,1,1,1,1,1,1], dtype=np.float32)
# Powerball AU suele ser 1-20; los otros 1-35 (ajusta si corresponde)
COL_MAX = np.array([35,35,35,35,35,35,35,20], dtype=np.float32)

# Intenta cargar modelo una sola vez
_model = None
_model_loaded = False
_model_error = None

try:
    if os.path.exists(MODEL_PATH):
        _model = tf.keras.models.load_model(MODEL_PATH)
        _model_loaded = True
    else:
        _model_error = f"File not found: {MODEL_PATH}"
except Exception as e:
    _model_error = str(e)
    _model_loaded = False

def is_model_loaded():
    return _model_loaded, _model_error

def minmax_scale(a, cmin, cmax):
    return (a - cmin) / (cmax - cmin + 1e-8)

def minmax_inverse(a_scaled, cmin, cmax):
    return a_scaled * (cmax - cmin + 1e-8) + cmin

def postprocess_prediction(vec_float):
    # redondea y limita por rango
    pred = np.rint(vec_float).astype(int)
    pred = np.clip(pred, COL_MIN, COL_MAX)

    main = pred[:7].tolist()
    pb = int(pred[7])

    # Mantener orden original (no ordenar)
    # Evitamos duplicados básicos desplazando hacia arriba si se repiten
    seen = set()
    for i in range(len(main)):
        while main[i] in seen and main[i] < int(COL_MAX[i]):
            main[i] += 1
        seen.add(main[i])

    return main, pb

def _get_last_draw_from_csv_or_fake():
    """
    Intenta leer el último sorteo de historical_draws.csv si existe.
    Si no existe, devuelve un fallback fijo (pero indicará que es fallback).
    historical_draws.csv debe tener columnas Numero1..Numero7,Powerball
    """
    if os.path.exists(HIST_CSV):
        import pandas as pd
        df = pd.read_csv(HIST_CSV)
        if not df.empty:
            last = df.iloc[-1][[f"Numero{i}" for i in range(1,8)] + ["Powerball"]].values.astype(np.float32)
            return last.reshape(1, -1), "csv"
    # fallback hardcoded (will show in /status)
    fake = np.array([[5,12,18,22,27,31,34,7]], dtype=np.float32)
    return fake, "fake"

def predict_from_last_draw():
    """
    Devuelve (nums_list, pb, meta) donde meta indica si viene de 'model' o 'fallback' y reasons
    """
    loaded, error = is_model_loaded()
    input_arr, input_source = _get_last_draw_from_csv_or_fake()

    if not loaded or _model is None:
        # No modelo: devolvemos el resultado del fallback (pero aún hacemos postprocessing con el input)
        out_real = input_arr[0]  # no model transformation; inverse scale no aplicable
        nums, pb = postprocess_prediction(out_real)
        return {"numbers": nums, "powerball": pb, "source": "fallback_input", "model_loaded": False, "model_error": error, "input_source": input_source}

    # escalar usando rangos asumidos (si en el futuro guardas col_min/col_max en entrenamiento, mejor)
    arr_sc = minmax_scale(input_arr, COL_MIN, COL_MAX)
    out_sc = _model.predict(arr_sc, verbose=0)
    out_real = minmax_inverse(out_sc, COL_MIN, COL_MAX)[0]
    nums, pb = postprocess_prediction(out_real)
    return {"numbers": nums, "powerball": pb, "source": "model", "model_loaded": True, "input_source": input_source}
