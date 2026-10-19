import os
import urllib.request

def descargar_dataset():
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv" # O el enlace/API de Kaggle
    ruta_destino = "datos/datos_ini/creditcard.csv"
    
    if not os.path.exists("datos/datos_ini"):
        os.makedirs("datos/datos_ini")
        
    print("Descargando dataset...")
    # Instrucción para Kaggle API o descarga directa
    print("Por favor descarga 'creditcard.csv' desde Kaggle y colócalo en datos/datos_ini/")

if __name__ == "__main__":
    descargar_dataset()