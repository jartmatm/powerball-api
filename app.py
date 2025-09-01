from fastapi import FastAPI
import numpy as np
import tensorflow as tf

# ====== Instancia FastAPI ======
app = FastAPI(title="Powerball Predictor API",
              description="API que sugiere números de Powerball basados en un modelo de IA",
              version="1.0")

# ====== Cargar modelo entrenado ======
# Asegúrate de que 'modelo_powerball.h5' esté en la misma carpeta que app.py
model = tf.keras.models.load_model("modelo_powerball.h5")

# ====== Endpoint raíz ======
@app.get("/")
def root():
    return {"message": "Bienvenido a la API Powerball 🎰"}

# ====== Endpoint de predicción ======
@app.post("/predict/")
def predict(input_numbers: list[int]):
    """
    Recibe 7 números + Powerball y devuelve predicción sugerida
    """
    if len(input_numbers) != 8:
        return {"error": "Debes enviar 8 números: 7 principales + Powerball"}

    try:
        # Convertir a numpy array
        x_input = np.array(input_numbers).reshape(1, -1).astype(np.float32)

        # Predicción usando el modelo
        prediction = model.predict(x_input, verbose=0)

        # Convertir a lista normal para JSON
        result = prediction[0].tolist()

        return {
            "input": input_numbers,
            "suggested_numbers": result
        }

    except Exception as e:
        return {"error": str(e)}

# Nota: NO necesitamos uvicorn.run aquí para Render
# El Start Command en Render será:
# uvicorn app:app --host 0.0.0.0 --port $PORT
