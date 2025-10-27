import warnings

warnings.filterwarnings("ignore")


import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

# 🔧 Parámetro: número de descargas simultáneas
MAX_WORKERS = 5
TIMEOUT = 200  # segundos

# 📂 Archivos de entrada
DATA_FILE = "nombres_archivos.csv"
LOG_FILE = "download_log.txt"
DOWNLOAD_DIR = "downloads"

# Crear carpeta de destino si no existe
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Leer el DataFrame desde el TXT
try:
    df_result = pd.read_csv(
        DATA_FILE, sep=",", encoding="utf-8"
    )  # usa utf-8 para evitar errores de caracteres
except Exception as e:
    print(f"❌ Error al leer '{DATA_FILE}': {e}")
    exit(1)

# Verificar que existan columnas necesarias
required_cols = {"file_name", "clean_link"}
if not required_cols.issubset(df_result.columns):
    print(f"❌ El archivo debe contener las columnas: {required_cols}")
    exit(1)

# Leer el índice de reanudación
last_index = 0
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as log:
        log_data = log.read().strip()
        if log_data.isdigit():
            last_index = int(log_data)
            print(f"🔄 Reanudando desde el índice {last_index}")
        else:
            print("⚠️ Log mal formado. Iniciando desde cero.")
else:
    print("🆕 No se encontró log. Iniciando desde cero.")

# Subconjunto pendiente por descargar
total_archivos = len(df_result)
df_to_download = df_result.iloc[last_index:]


# Función de descarga
def download_file(index, row):
    file_name = row["file_name"]
    file_path = os.path.join(DOWNLOAD_DIR, file_name)
    url = row["clean_link"]

    try:
        response = requests.get(url, verify=False, timeout=TIMEOUT)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            return index, True, None
        else:
            return index, False, f"HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        return index, False, str(e)


# Descarga paralela con barra de progreso
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(download_file, index, row): index
        for index, row in df_to_download.iterrows()
    }

    for future in tqdm(as_completed(futures), total=len(futures), desc="📥 Descargando archivos"):
        index, success, error = future.result()
        if success:
            # Guardar el índice del siguiente archivo
            with open(LOG_FILE, "w") as log:
                log.write(str(index + 1))
        else:
            print(f"⚠️ Error en índice {index}: {error}")
