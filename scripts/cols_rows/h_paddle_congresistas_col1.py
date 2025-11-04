import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from paddleocr import PaddleOCR
from tqdm import tqdm

# Directorios de entrada y salida
TIPO = "paddle_congresistas"
COLUMN_NAME = "columna_1"
DIR_NAMES_CSV = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/dir_names.csv"
PARQUET = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/congresistas_images.parquet"
OUTPUT_DIR = (
    f"/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/{TIPO}_{COLUMN_NAME}"
)

# Configuración de PaddleOCR
USE_GPU = True  # Cambiar a False si no tienes GPU con CUDA
LANG = "es"  # Español
BATCH_SIZE = 128  # Número de imágenes a procesar en lote (ajustar según VRAM disponible)

# Caracteres permitidos para el reconocimiento
ALLOWED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúñÁÉÍÓÚÑ,; "

# Variable global para el reader de PaddleOCR (se inicializa una sola vez)
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
    Inicializa el reader de PaddleOCR una sola vez.
    Esto es importante porque la inicialización carga los modelos en GPU/memoria.
    """
    global reader
    if reader is None:
        print(f"\n🔧 Inicializando PaddleOCR...")
        print(f"   - GPU habilitada: {USE_GPU}")
        print(f"   - Idioma: {LANG}")
        print(f"   - Caracteres permitidos: {len(ALLOWED_CHARS)} caracteres")
        reader = PaddleOCR(
            use_angle_cls=True,
            lang=LANG,
            use_gpu=USE_GPU,
            show_log=False,
            rec_char_whitelist=ALLOWED_CHARS,  # Limitar caracteres a reconocer
        )
        print("✓ PaddleOCR inicializado correctamente")
    return reader


def apply_ocr_to_image(image_path, ocr_reader):
    """
    Aplica PaddleOCR a una imagen y retorna el texto extraído.
    Retorna string vacío si la imagen no existe o hay error.

    Args:
        image_path: Ruta a la imagen
        ocr_reader: Instancia de PaddleOCR
    """
    try:
        if not os.path.exists(image_path):
            return ""

        # Aplicar OCR con PaddleOCR
        # ocr() retorna una lista de resultados por página
        # Cada resultado es una lista de líneas detectadas
        # Cada línea es: [bbox, (text, confidence)]
        result = ocr_reader.ocr(image_path, cls=True)

        if result is None or len(result) == 0 or result[0] is None:
            return ""

        # Extraer todos los textos detectados
        texts = []
        for line in result[0]:
            if line:
                text = line[1][0]  # line[1] es (text, confidence), line[1][0] es el texto
                if text.strip():  # Solo agregar si hay texto
                    texts.append(text)

        # Unir todos los textos detectados en una sola línea (con espacios)
        return " ".join(texts).strip()
    except Exception as e:
        return ""


def process_images_in_batches(image_paths, ocr_reader, batch_size):
    """
    Procesa múltiples imágenes en lotes para mayor eficiencia.

    Args:
        image_paths: Lista de rutas de imágenes
        ocr_reader: Instancia de PaddleOCR
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
    3. Aplica OCR a cada imagen usando PaddleOCR con GPU
    4. PaddleOCR reconoce solo: letras (a-z, A-Z), acentos, ñ, comas, punto y coma y espacios
    5. Agrupa resultados por documento y página
    6. Guarda JSONs
    """
    print("=" * 70)
    print("🚀 INICIANDO PROCESAMIENTO DE OCR EN COLUMNAS (PaddleOCR + GPU)")
    print("=" * 70)

    # Inicializar PaddleOCR
    ocr_reader = initialize_reader()

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
    print("\n🔬 5. Procesando OCR en imágenes con PaddleOCR...")
    print(f"⚙️  Procesamiento en GPU (batch_size={BATCH_SIZE})")
    print(
        f"🔤 PaddleOCR configurado para reconocer solo: letras (a-z, A-Z), acentos, ñ, comas y punto y coma"
    )
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
