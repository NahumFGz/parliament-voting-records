import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image
from tqdm import tqdm

# Directorios de entrada y salida
COLUMN_NAME = "columna_2"
DIR_NAMES_CSV = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/dir_names.csv"
CONGRESISTAS_PARQUET = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/congresistas_images.parquet"
OUTPUT_DIR = f"/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/{COLUMN_NAME}"

# Configuración de paralelización
NUM_WORKERS = 7  # Número de procesos paralelos (ajustar según CPU disponibles)


def natural_sort_key(text):
    """
    Clave para ordenamiento natural (fil_1.png, fil_2.png, ..., fil_10.png)
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r"(\d+)", text)]


def extract_document_id_and_page(dir_name):
    """
    Extrae el UUID del documento y el número de página del dir_name.
    Ejemplo: '000058a7-4618-53af-82f8-13266eef3ace_page003_' ->
             ('000058a7-4618-53af-82f8-13266eef3ace', 'page003')
    """
    # Patrón: UUID_pageXXX_
    match = re.match(r"^([a-f0-9\-]+)_(page\d+)_$", dir_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def apply_ocr_to_image(image_path):
    """
    Aplica pytesseract a una imagen y retorna el texto extraído.
    Retorna string vacío si la imagen no existe o hay error.
    """
    try:
        if not os.path.exists(image_path):
            return ""

        # Abrir imagen y aplicar OCR
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="spa")  # Usar español
        return text.strip()
    except Exception as e:
        return ""


def process_image_with_index(args):
    """
    Función auxiliar para procesamiento paralelo.
    Recibe una tupla (índice, image_path) y retorna (índice, texto).
    """
    index, image_path = args
    text = apply_ocr_to_image(image_path)
    return index, text


def get_processed_document_ids(output_dir):
    """
    Lee los archivos JSON existentes en el directorio de salida
    y retorna un set con los IDs de documentos ya procesados.
    """
    processed_ids = set()

    if not os.path.exists(output_dir):
        return processed_ids

    # Buscar todos los archivos .json en el directorio
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            # El nombre del archivo es el UUID del documento
            doc_id = filename.replace(".json", "")
            processed_ids.add(doc_id)

    return processed_ids


def process_images():
    """
    Procesa las imágenes:
    1. Lee dir_names.csv
    2. Filtra congresistas_images.parquet
    3. Detecta documentos ya procesados
    4. Aplica OCR a cada imagen de documentos pendientes
    5. Agrupa resultados por documento y página
    6. Guarda JSONs
    """
    print("=" * 70)
    print("🚀 INICIANDO PROCESAMIENTO DE OCR EN COLUMNAS")
    print("=" * 70)

    # 1. Leer dir_names.csv
    print("\n📂 1. Leyendo dir_names.csv...")
    if not os.path.exists(DIR_NAMES_CSV):
        print(f"❌ ERROR: No se encuentra {DIR_NAMES_CSV}")
        return

    dir_names_df = pd.read_csv(DIR_NAMES_CSV)
    dir_names_set = set(dir_names_df["dir_name"].tolist())
    print(f"✓ Se encontraron {len(dir_names_set)} dir_names únicos")

    # 2. Leer y filtrar parquet
    print("\n📂 2. Leyendo y filtrando congresistas_images.parquet...")
    if not os.path.exists(CONGRESISTAS_PARQUET):
        print(f"❌ ERROR: No se encuentra {CONGRESISTAS_PARQUET}")
        return

    df = pd.read_parquet(CONGRESISTAS_PARQUET)
    print(f"✓ Archivo parquet cargado: {len(df)} registros totales")

    # Filtrar por dir_names y column=COLUMN_NAME
    df_filtered = df[df["dir_name"].isin(dir_names_set) & (df["column"] == COLUMN_NAME)]
    print(f"✓ Después de filtrar por dir_names y {COLUMN_NAME}: {len(df_filtered)} registros")

    if len(df_filtered) == 0:
        print("⚠️  No hay registros para procesar después de filtrar")
        return

    # 3. Crear directorio de salida
    print(f"\n📁 3. Creando directorio de salida...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Directorio de salida: {OUTPUT_DIR}")

    # 4. Detectar documentos ya procesados
    print("\n🔍 4. Detectando documentos ya procesados...")
    processed_doc_ids = get_processed_document_ids(OUTPUT_DIR)
    print(f"✓ Se encontraron {len(processed_doc_ids)} documentos ya procesados")

    # 5. Agrupar por documento
    print("\n🔍 5. Agrupando datos por documento...")
    documents = defaultdict(lambda: defaultdict(list))

    # Agrupar las filas por documento y página
    for _, row in df_filtered.iterrows():
        dir_name = row["dir_name"]
        image_path = row["image_path"]
        image_name = row["image_name"]

        doc_id, page = extract_document_id_and_page(dir_name)
        if doc_id and page:
            # Solo agregar si el documento NO ha sido procesado
            if doc_id not in processed_doc_ids:
                documents[doc_id][page].append((image_path, image_name))

    total_documents = len(documents) + len(processed_doc_ids)
    print(f"✓ Documentos totales: {total_documents}")
    print(f"✓ Documentos ya procesados: {len(processed_doc_ids)}")
    print(f"✓ Documentos pendientes: {len(documents)}")

    if len(documents) == 0:
        print("\n🎉 ¡Todos los documentos ya han sido procesados!")
        print(f"📊 Resumen:")
        print(f"   - Documentos procesados: {len(processed_doc_ids)}")
        print(f"   - JSONs en: {OUTPUT_DIR}")
        print("=" * 70)
        return

    print(f"✓ Total de páginas a procesar: {sum(len(pages) for pages in documents.values())}")

    # 6. Procesar cada documento
    print("\n🔬 6. Procesando OCR en imágenes...")
    print(f"⚙️  Usando {NUM_WORKERS} workers en paralelo")

    # Calcular total de imágenes pendientes
    total_images = sum(len(images) for pages in documents.values() for images in pages.values())

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        with tqdm(total=total_images, desc="Procesando imágenes", unit="img") as pbar:
            for doc_id, pages in documents.items():
                doc_result = {}

                # Procesar cada página del documento
                for page, images in sorted(pages.items()):
                    # Ordenar imágenes por nombre usando ordenamiento natural
                    images_sorted = sorted(images, key=lambda x: natural_sort_key(x[1]))

                    # Preparar argumentos para procesamiento paralelo
                    tasks = [(i, img_path) for i, (img_path, _) in enumerate(images_sorted)]

                    # Enviar tareas al pool de workers
                    futures = {
                        executor.submit(process_image_with_index, task): task for task in tasks
                    }

                    # Recolectar resultados manteniendo el orden
                    results = {}
                    for future in as_completed(futures):
                        index, text = future.result()
                        results[index] = text
                        pbar.update(1)

                    # Ordenar resultados por índice
                    page_texts = [results[i] for i in range(len(results))]
                    doc_result[page] = page_texts

                # Guardar JSON del documento
                json_filename = f"{doc_id}.json"
                json_path = os.path.join(OUTPUT_DIR, json_filename)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ ¡Proceso completado!")
    print(f"📊 Resumen:")
    print(f"   - Documentos procesados en esta ejecución: {len(documents)}")
    print(f"   - Documentos ya existentes: {len(processed_doc_ids)}")
    print(f"   - Total de documentos: {len(documents) + len(processed_doc_ids)}")
    print(f"   - Imágenes procesadas en esta ejecución: {total_images}")
    print(f"   - JSONs guardados en: {OUTPUT_DIR}")
    print("=" * 70)


def main():
    process_images()


if __name__ == "__main__":
    main()
