import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import easyocr
import pandas as pd
import pytesseract
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from paddleocr import PaddleOCR
from tqdm import tqdm

# Directorios de entrada y salida
TIPO = "congresista"
COLUMN_NAME = "columna_3"
DIR_NAMES_CSV = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/dir_names.csv"
PARQUET = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/congresistas_images.parquet"
OUTPUT_DIR = (
    f"/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows/{TIPO}_{COLUMN_NAME}"
)

# Configuración de OCR
USE_GPU = True
LANGUAGES = ["es"]
MODEL_PRIORITY = ["Tesseract", "PaddleOCR", "docTR", "EasyOCR"]

# Variables globales para los readers de OCR (se inicializan una sola vez)
easyocr_reader = None
doctr_reader = None
paddleocr_reader = None
# Tesseract no necesita reader global, usa pytesseract directamente


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


def initialize_readers():
    """
    Inicializa los readers de OCR según MODEL_PRIORITY.
    Esto es importante porque la inicialización carga los modelos en GPU/memoria.
    """
    global easyocr_reader, doctr_reader, paddleocr_reader

    print(f"\n🔧 Inicializando modelos OCR...")
    print(f"   - GPU habilitada: {USE_GPU}")
    print(f"   - Idiomas: {LANGUAGES}")
    print(f"   - Prioridad de modelos: {MODEL_PRIORITY}")

    for model_name in MODEL_PRIORITY:
        try:
            if model_name == "EasyOCR" and easyocr_reader is None:
                print(f"   ⏳ Inicializando EasyOCR...")
                easyocr_reader = easyocr.Reader(LANGUAGES, gpu=USE_GPU, verbose=False)
                print(f"   ✓ EasyOCR inicializado correctamente")

            elif model_name == "docTR" and doctr_reader is None:
                print(f"   ⏳ Inicializando docTR...")
                doctr_reader = ocr_predictor(
                    det_arch="db_resnet50",
                    reco_arch="crnn_vgg16_bn",
                    pretrained=True,
                )
                print(f"   ✓ docTR inicializado correctamente")

            elif model_name == "PaddleOCR" and paddleocr_reader is None:
                print(f"   ⏳ Inicializando PaddleOCR...")
                paddleocr_reader = PaddleOCR(
                    use_angle_cls=True,
                    lang="es",
                    use_gpu=USE_GPU,
                    show_log=False,
                )
                print(f"   ✓ PaddleOCR inicializado correctamente")

            elif model_name == "Tesseract":
                print(f"   ✓ Tesseract disponible (no requiere inicialización)")

        except Exception as e:
            print(f"   ⚠️  Error inicializando {model_name}: {e}")

    print("✓ Modelos OCR inicializados")


def apply_easyocr(image_path):
    """Aplica EasyOCR a una imagen."""
    try:
        if easyocr_reader is None:
            return ""
        result = easyocr_reader.readtext(image_path, detail=0)
        text = "\n".join(result)
        return text.strip()
    except Exception as e:
        return ""


def apply_doctr(image_path):
    """Aplica docTR a una imagen."""
    try:
        if doctr_reader is None:
            return ""
        doc = DocumentFile.from_images(image_path)
        result = doctr_reader(doc)

        # Extraer texto de los resultados
        text_lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    text_lines.append(line_text)

        return "\n".join(text_lines).strip()
    except Exception as e:
        return ""


def apply_paddleocr(image_path):
    """Aplica PaddleOCR a una imagen."""
    try:
        if paddleocr_reader is None:
            return ""
        result = paddleocr_reader.ocr(image_path, cls=True)

        # Extraer texto de los resultados
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) > 1:
                    text_lines.append(line[1][0])

        return "\n".join(text_lines).strip()
    except Exception as e:
        return ""


def apply_tesseract(image_path):
    """Aplica Tesseract a una imagen."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
        text = pytesseract.image_to_string(img, lang="spa")
        return text.strip()
    except Exception as e:
        return ""


def apply_ocr_to_image(image_path):
    """
    Aplica OCR a una imagen usando múltiples modelos según MODEL_PRIORITY.
    Retorna el texto del primer modelo que devuelva resultado no vacío.
    Retorna string vacío si ningún modelo devuelve resultado.

    Args:
        image_path: Ruta a la imagen
    """
    if not os.path.exists(image_path):
        return ""

    # Intentar cada modelo en el orden especificado
    for model_name in MODEL_PRIORITY:
        try:
            text = ""

            if model_name == "EasyOCR":
                text = apply_easyocr(image_path)
            elif model_name == "docTR":
                text = apply_doctr(image_path)
            elif model_name == "PaddleOCR":
                text = apply_paddleocr(image_path)
            elif model_name == "Tesseract":
                text = apply_tesseract(image_path)

            # Si encontramos texto, retornar inmediatamente
            if text:
                return text

        except Exception as e:
            # Si hay error, continuar con el siguiente modelo
            continue

    # Si ningún modelo devolvió resultado, retornar vacío
    return ""


def process_images():
    """
    Procesa las imágenes:
    1. Lee dir_names.csv
    2. Filtra congresistas_images.parquet
    3. Aplica OCR a cada imagen usando múltiples modelos con fallback
    4. Agrupa resultados por documento y página
    5. Guarda JSONs
    """
    print("=" * 70)
    print("🚀 INICIANDO PROCESAMIENTO DE OCR EN COLUMNAS (Multi-Modelo + GPU)")
    print("=" * 70)

    # Inicializar modelos OCR
    initialize_readers()

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
    print("\n🔬 5. Procesando OCR en imágenes con múltiples modelos...")
    print(f"⚙️  Procesamiento secuencial en GPU")
    print(f"⚙️  Prioridad de modelos: {' → '.join(MODEL_PRIORITY)}")
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

                # Procesar imágenes secuencialmente
                page_texts = []
                for img_path in image_paths:
                    text = apply_ocr_to_image(img_path)
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
