import os

from PIL import Image

input_path = "../data/b_originales"
output_path = "../data/c_comprimidos"

# Crear carpeta de salida si no existe
os.makedirs(output_path, exist_ok=True)

# Parámetros de compresión
quality = 85  # 0-100 (recomendado 80–90 para OCR)
optimize = True

for file_name in os.listdir(input_path):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")):
        input_file = os.path.join(input_path, file_name)
        output_file = os.path.join(output_path, file_name)

        # Abrir y convertir a escala de grises
        with Image.open(input_file) as img:
            gray = img.convert("L")  # Escala de grises

            # Convertir PNGs a JPEG para mejor compresión (opcional)
            if file_name.lower().endswith(".png"):
                output_file = os.path.splitext(output_file)[0] + ".jpg"

            # Guardar con compresión
            gray.save(output_file, format="JPEG", quality=quality, optimize=optimize)

        print(f"Procesada: {file_name} → {output_file}")

print("✅ Conversión y compresión completadas.")
