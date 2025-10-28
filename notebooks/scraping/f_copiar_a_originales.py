import os
import shutil
from pathlib import Path

ORIGINALES_INPUT_PATH = "../../data/scraping/c_classified/votacion"
ORIGINALES_OUTPUT_PATH = "../../data/procesamiento_todas_votaciones/a_originales"


# Función para copiar imágenes de una carpeta a otra
def copy_images(input_path, output_path):
    """
    Copia todos los archivos de imagen desde input_path a output_path
    """
    input_dir = Path(input_path)
    output_dir = Path(output_path)

    # Crear el directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verificar que el directorio de entrada existe
    if not input_dir.exists():
        print(f"⚠️  El directorio {input_dir} no existe. Saltando...")
        return

    # Extensiones de imagen soportadas
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}

    # Contadores
    copied_count = 0
    error_count = 0

    # Copiar todos los archivos de imagen
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                dest_path = output_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_count += 1
                print(f"✓ Copiado: {file_path.name}")
            except Exception as e:
                error_count += 1
                print(f"✗ Error copiando {file_path.name}: {e}")

    print(f"\n📊 Resultado para {input_path}:")
    print(f"  - Copiados: {copied_count}")
    print(f"  - Errores: {error_count}")


# Copiar imágenes para cada tipo
print("=" * 60)
print("Copiando ORIGINALES...")
print("=" * 60)

copy_images(ORIGINALES_INPUT_PATH, ORIGINALES_OUTPUT_PATH)

print("\n" + "=" * 60)
print("✅ Proceso completado")
print("=" * 60)
