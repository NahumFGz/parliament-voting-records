IMAGE_CSV_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/scripts/list_images_ocr_openai/carpetas_validas.csv"
OUTPUT_DIR = "/home/nahumfg/GithubProjects/parliament-voting-records/extract_ocr/encabezados"
COL_NAME = "ENCABEZADO"
NUM_WORKERS = 10  # Número de trabajadores paralelos
MAX_RETRIES = 3  # Número máximo de reintentos por imagen
RETRY_DELAY_BASE = 5  # Segundos de espera base entre reintentos (se multiplica exponencialmente)

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from utils_openai_ocr import process_image_ocr

# Lock para escritura segura en consola
print_lock = Lock()


def safe_print(message):
    """Imprime de forma segura en entorno multi-thread"""
    with print_lock:
        print(message)


def procesar_imagen(row_data, idx, total):
    """
    Procesa una imagen individual con reintentos

    Args:
        row_data: tupla con (dir_path, dir_name, col_name_value)
        idx: índice actual
        total: total de imágenes a procesar

    Returns:
        tuple: (success: bool, dir_name: str, error_msg: str or None)
    """
    dir_path, dir_name, col_name_value = row_data

    # Construir la ruta completa de la imagen
    image_path = os.path.join(dir_path, col_name_value)

    # Construir la ruta de salida
    output_path = os.path.join(OUTPUT_DIR, f"{dir_name}.json")

    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        safe_print(f"[{idx}/{total}] SALTADO - No existe: {image_path}")
        return (False, dir_name, "Archivo no existe")

    # Intentar procesar con reintentos
    for intento in range(1, MAX_RETRIES + 1):
        try:
            safe_print(f"[{idx}/{total}] Procesando: {dir_name} (intento {intento}/{MAX_RETRIES})")

            result = process_image_ocr(
                image_path=image_path,
                resize_percent=90,
                model="gpt-5-mini",
                max_tokens=2500,
                prompt="En base a la imagen extrae un json con el tipo (Asistencia o Votación), fecha, hora, presidente y asunto. Solo quiero el json sin comentarios adicionales. Si no encuentras alguno de esos campos pon None",
                output_path=output_path,
            )

            safe_print(f"[{idx}/{total}] ✓ {dir_name} - Guardado exitosamente")
            return (True, dir_name, None)

        except Exception as e:
            error_msg = str(e)
            if intento < MAX_RETRIES:
                delay = RETRY_DELAY_BASE * (2 ** (intento - 1))  # Backoff exponencial: 5s, 10s, 20s
                safe_print(
                    f"[{idx}/{total}] ⚠ {dir_name} - Error (reintentando en {delay}s): {error_msg}"
                )
                time.sleep(delay)
            else:
                safe_print(
                    f"[{idx}/{total}] ✗ {dir_name} - Error final después de {MAX_RETRIES} intentos: {error_msg}"
                )
                return (False, dir_name, error_msg)

    return (False, dir_name, "Número máximo de reintentos alcanzado")


# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Leer el CSV
df = pd.read_csv(IMAGE_CSV_PATH)

# Obtener lista de archivos JSON ya procesados
archivos_procesados = set()
if os.path.exists(OUTPUT_DIR):
    for archivo in os.listdir(OUTPUT_DIR):
        if archivo.endswith(".json"):
            # Extraer el nombre sin la extensión .json
            nombre_sin_extension = archivo[:-5]
            archivos_procesados.add(nombre_sin_extension)

print(f"Se encontraron {len(archivos_procesados)} archivos ya procesados")
print(f"Total de filas en CSV: {len(df)}")

# Filtrar el DataFrame para excluir los ya procesados
df_filtrado = df[~df["DIR_NAME"].isin(archivos_procesados)]

print(f"Filas por procesar: {len(df_filtrado)}")
print(f"Trabajadores paralelos: {NUM_WORKERS}")
print("-" * 60)

# Preparar datos para procesamiento paralelo
tareas = []
for idx, (index, row) in enumerate(df_filtrado.iterrows(), 1):
    row_data = (row["DIR_PATH"], row["DIR_NAME"], row[COL_NAME])
    tareas.append((row_data, idx, len(df_filtrado)))

# Procesar en paralelo con ThreadPoolExecutor
start_time = time.time()
exitosos = 0
fallidos = 0
errores = []

print(f"\n🚀 Iniciando procesamiento paralelo...\n")

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    # Enviar todas las tareas
    futures = {executor.submit(procesar_imagen, *tarea): tarea for tarea in tareas}

    # Procesar resultados conforme se completan
    for future in as_completed(futures):
        try:
            success, dir_name, error_msg = future.result()
            if success:
                exitosos += 1
            else:
                fallidos += 1
                if error_msg:
                    errores.append((dir_name, error_msg))
        except Exception as e:
            fallidos += 1
            safe_print(f"✗ Error inesperado en thread: {str(e)}")

# Resumen final
elapsed_time = time.time() - start_time
total_procesados = exitosos + fallidos

print("\n" + "=" * 60)
print("RESUMEN DE PROCESAMIENTO")
print("=" * 60)
print(f"Total procesados: {total_procesados}")
print(f"✓ Exitosos: {exitosos}")
print(f"✗ Fallidos: {fallidos}")
print(f"⏱ Tiempo total: {elapsed_time:.2f} segundos")
if total_procesados > 0:
    print(f"⚡ Promedio: {elapsed_time/total_procesados:.2f} seg/imagen")
print("=" * 60)

# Mostrar errores si hay
if errores:
    print("\nERRORES ENCONTRADOS:")
    print("-" * 60)
    for dir_name, error_msg in errores[:10]:  # Mostrar máximo 10 errores
        print(f"  • {dir_name}: {error_msg}")
    if len(errores) > 10:
        print(f"  ... y {len(errores) - 10} errores más")
    print("-" * 60)
