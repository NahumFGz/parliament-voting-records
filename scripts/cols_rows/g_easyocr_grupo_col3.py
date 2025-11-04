import json
import os
import re
from collections import defaultdict
from pathlib import Path

import easyocr
import pandas as pd
from tqdm import tqdm

# Directorios de entrada y salida
TIPO = "grupo_parlamentario"
COLUMN_NAME = "columna_3"
DIR_NAMES_CSV = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/dir_names.csv"
PARQUET = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/grupo_parlamentario_images.parquet"
OUTPUT_DIR = (
    f"/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/{TIPO}_{COLUMN_NAME}"
)

# Configuración de EasyOCR
USE_GPU = True  # Cambiar a False si no tienes GPU con CUDA
LANGUAGES = ["es"]  # Español
BATCH_SIZE = 1024  # Número de imágenes a procesar en lote (ajustar según VRAM disponible)

# Variable global para el reader de EasyOCR (se inicializa una sola vez)
reader = None


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


def initialize_reader():
    """
    Inicializa el reader de EasyOCR una sola vez.
    Esto es importante porque la inicialización carga los modelos en GPU/memoria.
    """
    global reader
    if reader is None:
        print(f"\n🔧 Inicializando EasyOCR...")
        print(f"   - GPU habilitada: {USE_GPU}")
        print(f"   - Idiomas: {LANGUAGES}")
        reader = easyocr.Reader(LANGUAGES, gpu=USE_GPU, verbose=False)
        print("✓ EasyOCR inicializado correctamente")
    return reader


def apply_ocr_to_image(image_path, ocr_reader):
    """
    Aplica EasyOCR a una imagen y retorna el texto extraído.
    Retorna string vacío si la imagen no existe o hay error.

    Args:
        image_path: Ruta a la imagen
        ocr_reader: Instancia de easyocr.Reader
    """
    try:
        if not os.path.exists(image_path):
            return ""

        # Aplicar OCR con EasyOCR
        # readtext retorna una lista de tuplas: (bbox, text, confidence)
        result = ocr_reader.readtext(image_path, detail=0)  # detail=0 retorna solo texto

        # Unir todos los textos detectados con saltos de línea
        text = "\n".join(result)
        return text.strip()
    except Exception as e:
        return ""


def process_images_in_batches(image_paths, ocr_reader, batch_size):
    """
    Procesa múltiples imágenes en lotes para mayor eficiencia.

    Args:
        image_paths: Lista de rutas de imágenes
        ocr_reader: Instancia de easyocr.Reader
        batch_size: Tamaño del lote

    Returns:
        Lista de textos extraídos en el mismo orden que image_paths
    """
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]

        for img_path in batch:
            text = apply_ocr_to_image(img_path, ocr_reader)
            results.append(text)

    return results


def process_images():
    """
    Procesa las imágenes:
    1. Lee dir_names.csv
    2. Filtra congresistas_images.parquet
    3. Aplica OCR a cada imagen usando EasyOCR con GPU
    4. Agrupa resultados por documento y página
    5. Guarda JSONs
    """
    print("=" * 70)
    print("🚀 INICIANDO PROCESAMIENTO DE OCR EN COLUMNAS (EasyOCR + GPU)")
    print("=" * 70)

    # Inicializar EasyOCR
    ocr_reader = initialize_reader()

    # 1. Leer dir_names.csv
    print("\n📂 1. Leyendo dir_names.csv...")
    if not os.path.exists(DIR_NAMES_CSV):
        print(f"❌ ERROR: No se encuentra {DIR_NAMES_CSV}")
        return

    dir_names_df = pd.read_csv(DIR_NAMES_CSV)
    print(f"✓ Se encontraron {len(dir_names_df)} dir_names totales")

    # Filtrar dir_names que ya fueron procesados
    # Los JSONs se guardan con el UUID (sin _pageXXX_), así que extraemos el UUID de cada dir_name
    existing_jsons = set()
    if os.path.exists(OUTPUT_DIR):
        existing_jsons = {
            f.replace(".json", "") for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")
        }
        print(f"✓ Se encontraron {len(existing_jsons)} documentos ya procesados")

    # Filtrar dir_names cuyos UUIDs ya existen
    def get_uuid_from_dirname(dir_name):
        """Extrae el UUID del dir_name (quita _pageXXX_)"""
        doc_id, _ = extract_document_id_and_page(dir_name)
        return doc_id

    dir_names_df["uuid"] = dir_names_df["dir_name"].apply(get_uuid_from_dirname)
    dir_names_df_filtered = dir_names_df[~dir_names_df["uuid"].isin(existing_jsons)]
    dir_names_set = set(dir_names_df_filtered["dir_name"].tolist())

    print(f"✓ Después de filtrar procesados: {len(dir_names_set)} dir_names a procesar")

    if len(dir_names_set) == 0:
        print("⚠️  No hay dir_names nuevos para procesar (todos ya fueron procesados)")
        return

    # 2. Leer y filtrar parquet
    print("\n📂 2. Leyendo y filtrando congresistas_images.parquet...")
    if not os.path.exists(PARQUET):
        print(f"❌ ERROR: No se encuentra {PARQUET}")
        return

    df = pd.read_parquet(PARQUET)
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

    # 4. Agrupar por documento
    print("\n🔍 4. Agrupando datos por documento...")
    documents = defaultdict(lambda: defaultdict(list))

    # Agrupar las filas por documento y página
    for _, row in df_filtered.iterrows():
        dir_name = row["dir_name"]
        image_path = row["image_path"]
        image_name = row["image_name"]

        doc_id, page = extract_document_id_and_page(dir_name)
        if doc_id and page:
            documents[doc_id][page].append((image_path, image_name))

    print(f"✓ Se encontraron {len(documents)} documentos únicos")
    print(f"✓ Total de páginas a procesar: {sum(len(pages) for pages in documents.values())}")

    # 5. Procesar cada documento
    print("\n🔬 5. Procesando OCR en imágenes con EasyOCR...")
    print(f"⚙️  Procesamiento en GPU (batch_size={BATCH_SIZE})")
    total_images = len(df_filtered)

    with tqdm(total=total_images, desc="Procesando imágenes", unit="img") as pbar:
        for doc_id, pages in documents.items():
            doc_result = {}

            # Procesar cada página del documento
            for page, images in sorted(pages.items()):
                # Ordenar imágenes por nombre usando ordenamiento natural
                images_sorted = sorted(images, key=lambda x: natural_sort_key(x[1]))

                # Extraer solo las rutas de imágenes
                image_paths = [img_path for img_path, _ in images_sorted]

                # Procesar imágenes en lotes
                page_texts = []
                for img_path in image_paths:
                    text = apply_ocr_to_image(img_path, ocr_reader)
                    page_texts.append(text)
                    pbar.update(1)

                doc_result[page] = page_texts

            # Guardar JSON del documento
            json_filename = f"{doc_id}.json"
            json_path = os.path.join(OUTPUT_DIR, json_filename)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc_result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ ¡Proceso completado!")
    print(f"📊 Resumen:")
    print(f"   - Documentos procesados: {len(documents)}")
    print(f"   - Imágenes procesadas: {total_images}")
    print(f"   - JSONs guardados en: {OUTPUT_DIR}")
    print("=" * 70)


def main():
    process_images()


if __name__ == "__main__":
    main()
