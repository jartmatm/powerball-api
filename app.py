from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# ====== Instancia FastAPI ======
app = FastAPI(
    title="Powerball Predictor API",
    description="API que sugiere números de Powerball basados en un modelo de IA",
    version="1.0"
)

# ====== Cargar modelo entrenado ======
# Asegúrate de que 'modelo_powerball.keras' esté en la misma carpeta que app.py
model = tf.keras.models.load_model("modelo_powerball.keras")

# ====== Modelo de entrada ======
class InputData(BaseModel):
    numbers: list[int]

# ====== Endpoint raíz ======
@app.get("/")
def root():
    return {"message": "Bienvenido a la API Powerball 🎰"}

# ====== Endpoint de predicción ======
@app.post("/predict/")
def predict(data: InputData):
    """
    Recibe 7 números + Powerball en un JSON y devuelve predicción sugerida
    """
    input_numbers = data.numbers

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



