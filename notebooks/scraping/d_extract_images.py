import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from pdf2image import convert_from_path

PDFS_FOLDER = "../../data/scraping/a_pdfs"
IMAGES_FOLDER = "../../data/scraping/b_extract_images"

# Número de procesos paralelos
NUM_WORKERS = 12  # 🔧 Ajusta según tus núcleos disponibles

os.makedirs(IMAGES_FOLDER, exist_ok=True)


def process_pdf(file: str):
    """Convierte un PDF en imágenes y devuelve un resumen del progreso."""
    pdf_path = os.path.join(PDFS_FOLDER, file)
    images = convert_from_path(pdf_path)
    total_pages = len(images)
    base_name = file.rsplit(".", 1)[0]

    for i, image in enumerate(images, start=1):
        page_str = f"_page{str(i)}_"
        image_filename = f"{base_name}{page_str}.png"
        image.save(os.path.join(IMAGES_FOLDER, image_filename))
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
