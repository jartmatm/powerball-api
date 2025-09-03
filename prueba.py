import requests

# URL de tu API local
url = "http://127.0.0.1:8000/predict_next/"

try:
    response = requests.get(url)
    response.raise_for_status()  # Lanza error si el status no es 200
    data = response.json()
    print("Predicción del siguiente sorteo:")
    print("Números sugeridos:", data.get("suggested_numbers"))
    print("Powerball sugerido:", data.get("suggested_powerball"))

except requests.exceptions.RequestException as e:
    print("Error al conectar con la API:", e)
except ValueError:
    print("Respuesta no es JSON válido:", response.text)
