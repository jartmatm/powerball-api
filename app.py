# app.py
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import modelo_powerball

app = FastAPI(
    title="Powerball Predictor API",
    description="API que sugiere números de Powerball basados en un modelo de IA",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "Bienvenido a la API Powerball 🎰"}

@app.get("/status")
def status():
    loaded, err = modelo_powerball.is_model_loaded()
    return {"model_loaded": loaded, "model_error": err}

@app.get("/predict_next/")
def predict_next_draw():
    """
    Llama a modelo_powerball.predict_from_last_draw() y devuelve JSON con meta info.
    """
    try:
        res = modelo_powerball.predict_from_last_draw()
        # res ya es dict con keys numbers,powerball,source,model_loaded,...
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
