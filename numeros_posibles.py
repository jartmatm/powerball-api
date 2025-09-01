import pandas as pd

df = pd.read_excel('C:/Users/jessi/OneDrive/Desktop/df_lotto_prueba_final.xlsx', index_col=None)

# Convierte todas las columnas de números a enteros con soporte NaN
for col in ["Numero1","Numero2","Numero3","Numero4","Numero5","Numero6","Numero7","Powerball"]:
    df[col] = df[col].astype("Int64")  # con mayúscula, permite NaN
print(df.head(14))

columnas_numeros = [col for col in df.columns if col.startswith("Numero") or col == "Powerball"]

resultados = []

for col in columnas_numeros:
    conteo = df[col].value_counts().head(5)
    for numero, frecuencia in conteo.items():
        # Limpiar comas y convertir a entero
        numero_int = int(str(numero).replace(",", ""))
        resultados.append({
            "Columna": col,
            "Numero": numero_int,
            "Frecuencia": int(frecuencia)
        })

top5_df = pd.DataFrame(resultados)
print(top5_df)
