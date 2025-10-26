import os

from PIL import Image

input_path = "../data/b_originales"
output_path = "../data/c_comprimidos"

# Crear carpeta de salida si no existe
os.makedirs(output_path, exist_ok=True)

# Parámetros de compresión
quality = 90  # Para documentos escaneados, 90 es mejor que 85
optimize = True
target_dpi = 300  # DPI estándar para OCR

for file_name in os.listdir(input_path):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")):
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

                print(
                    f"✓ {file_name} → {os.path.basename(output_file)} "
                    f"({original_size:.1f}KB → {compressed_size:.1f}KB, "
                    f"{reduction:.1f}% reducción)"
                )

        except Exception as e:
            print(f"✗ Error procesando {file_name}: {e}")

print("\n✅ Conversión y compresión completadas.")
