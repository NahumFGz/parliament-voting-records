import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from pdf2image import convert_from_path

PDFS_FOLDER = "../../data/scraping/a_pdfs"
IMAGES_FOLDER = "../../data/scraping/b_extract_images"

# Número de procesos paralelos
NUM_WORKERS = 12  # 🔧 Ajusta según tus núcleos disponibles

# Configuración de compresión y procesamiento de imágenes
DPI = 300  # 🔧 Resolución de extracción del PDF (200=básico, 300=estándar, 400+=alta calidad)
JPEG_QUALITY = 90  # 🔧 Calidad de compresión JPEG (1-100, donde 100 es máxima calidad)
CONVERT_TO_GRAYSCALE = True  # 🔧 True para convertir a escala de grises, False para mantener color

os.makedirs(IMAGES_FOLDER, exist_ok=True)


def process_pdf(file: str):
    """Convierte un PDF en imágenes y devuelve un resumen del progreso."""
    pdf_path = os.path.join(PDFS_FOLDER, file)
    images = convert_from_path(pdf_path, dpi=DPI)
    total_pages = len(images)
    base_name = file.rsplit(".", 1)[0]

    for i, image in enumerate(images, start=1):
        # Convertir a escala de grises si está configurado
        if CONVERT_TO_GRAYSCALE:
            image = image.convert("L")

        # Nombre de archivo con formato nombre_del_pdf_page001_.jpg
        image_filename = f"{base_name}_page{str(i).zfill(3)}_.jpg"
        image_path = os.path.join(IMAGES_FOLDER, image_filename)

        # Guardar con compresión JPEG
        image.save(image_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

        progress = (i / total_pages) * 100
        print(f"   🖼️ {file}: Página {i}/{total_pages} ({progress:.1f}%)")

    return f"{file} completado ({total_pages} páginas)"


# Lista de PDFs
pdf_files = [f for f in os.listdir(PDFS_FOLDER) if f.endswith(".pdf")]
total_pdfs = len(pdf_files)

print(f"📄 Se encontraron {total_pdfs} archivos PDF para procesar.")
print(f"🚀 Procesando en paralelo con {NUM_WORKERS} workers...\n")

# Ejecutar en paralelo
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_pdf, file): file for file in pdf_files}
    for idx, future in enumerate(as_completed(futures), start=1):
        file = futures[future]
        try:
            result = future.result()
            print(f"✅ [{idx}/{total_pdfs}] {result}")
        except Exception as e:
            print(f"❌ Error procesando {file}: {e}")

print("\n🎉 Todos los PDFs han sido procesados correctamente.")
