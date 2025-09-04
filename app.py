from fastapi import FastAPI
from modelo_powerball import predict_from_last_draw
import os
import uvicorn

# ====== Instancia FastAPI ======
app = FastAPI(
    title="Powerball Predictor API",
    description="API que sugiere números de Powerball basados en un modelo de IA",
    version="1.0"
)

# ====== Endpoint raíz ======
@app.get("/")
def root():
    return {"message": "Bienvenido a la API Powerball 🎰"}

# ====== Endpoint para predecir el siguiente sorteo ======
@app.get("/predict_next/")
def predict_next_draw():
    """
    Devuelve los números sugeridos para el siguiente sorteo,
    usando el último sorteo histórico.
    """
    try:
        nums, pb = predict_from_last_draw()
        return {
            "suggested_numbers": nums,
            "suggested_powerball": pb
        }
    except Exception as e:
        return {"error": str(e)}

# ====== Para correr local ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render asigna el puerto automáticamente
    uvicorn.run(app, host="0.0.0.0", port=port)
