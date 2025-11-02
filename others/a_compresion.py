import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

input_path = "../data/procesamiento_todas_votaciones/a_originales"
output_path = "../data/procesamiento_todas_votaciones/b_comprimidos"

# Crear carpeta de salida si no existe
os.makedirs(output_path, exist_ok=True)

# Parámetros de compresión
quality = 90
optimize = True
target_dpi = 300
max_workers = 14  # None = usar todos los CPUs disponibles


def process_image(args):
    """
    Procesa una sola imagen: convierte a escala de grises y comprime.
    Esta función será ejecutada en paralelo.

    Args:
        args: tupla con (file_name, input_path, output_path, quality, optimize, target_dpi)
    """
    file_name, input_path, output_path, quality, optimize, target_dpi = args

    if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")):
        return None

    input_file = os.path.join(input_path, file_name)
    output_file = os.path.join(output_path, file_name)

    try:
        with Image.open(input_file) as img:
            # Preservar DPI si existe, o establecer 300 DPI
            dpi = img.info.get("dpi", (target_dpi, target_dpi))

            # Convertir a escala de grises
            gray = img.convert("L")

            # Opcional: Aplicar threshold para binarización (blanco/negro puro)
            # Esto reduce aún más el tamaño y mejora OCR en docs B&N
            # Descomenta si quieres:
            # from PIL import ImageOps
            # gray = ImageOps.autocontrast(gray)
            # threshold = 128
            # gray = gray.point(lambda x: 0 if x < threshold else 255, '1')

            # Convertir PNGs a JPEG
            if file_name.lower().endswith(".png"):
                output_file = os.path.splitext(output_file)[0] + ".jpg"

            # Guardar con compresión y DPI
            gray.save(output_file, format="JPEG", quality=quality, optimize=optimize, dpi=dpi)

            # Reportar reducción de tamaño
            original_size = os.path.getsize(input_file) / 1024  # KB
            compressed_size = os.path.getsize(output_file) / 1024  # KB
            reduction = ((original_size - compressed_size) / original_size) * 100

            return {
                "success": True,
                "file_name": file_name,
                "output_file": os.path.basename(output_file),
                "original_size": original_size,
                "compressed_size": compressed_size,
                "reduction": reduction,
            }

    except Exception as e:
        return {
            "success": False,
            "file_name": file_name,
            "error": str(e),
        }


# Obtener lista de archivos a procesar
file_list = [
    f
    for f in os.listdir(input_path)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"))
]

print(
    f"📦 Procesando {len(file_list)} imágenes con {max_workers if max_workers else os.cpu_count()} workers...\n"
)

# Procesar imágenes en paralelo
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Preparar argumentos para cada tarea
    tasks = [
        (file_name, input_path, output_path, quality, optimize, target_dpi)
        for file_name in file_list
    ]

    # Enviar todas las tareas
    future_to_file = {executor.submit(process_image, task): task[0] for task in tasks}

    # Procesar resultados conforme se completan
    for future in as_completed(future_to_file):
        result = future.result()
        if result:
            if result["success"]:
                print(
                    f"✓ {result['file_name']} → {result['output_file']} "
                    f"({result['original_size']:.1f}KB → {result['compressed_size']:.1f}KB, "
                    f"{result['reduction']:.1f}% reducción)"
                )
            else:
                print(f"✗ Error procesando {result['file_name']}: {result['error']}")

print("\n✅ Conversión y compresión completadas.")
