import pandas as pd
import requests
from bs4 import BeautifulSoup
from url_years import urls

all_data = []

for year, url in urls.items():
    print(f'procesando{year}...')
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

df.to_excel('C:/Users/jessi/OneDrive/Desktop/df_lotto_prueba_final.xlsx', index=False)