import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

# ⚙️ Configuraciones
base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"

# 🔧 Configuración de paralelización
# Si es 0, procesa secuencialmente. Si > 0, usa ese número de workers
NUM_WORKERS = 8  # Puedes cambiar este valor según necesites


def procesar_imagen(args):
    """
    Procesa una imagen individual: divide en 4 cuadrantes y guarda el superior derecho.

    Args:
        args: Tupla con (idx, img_path, total_imgs)

    Returns:
        str: Mensaje de estado del procesamiento
    """
    idx, img_path, total_imgs = args

    if idx % 100 == 0 or idx == 1:
        print(f"\n📊 Progreso: {idx}/{total_imgs} imágenes procesadas")

    image_bgr = cv2.imread(img_path)

    if image_bgr is None:
        return f"[⚠️] No se pudo leer la imagen: {img_path}"

    # Obtener dimensiones de la imagen
    img_height, img_width = image_bgr.shape[:2]

    # Calcular el centro de la imagen
    centro_x = img_width // 2
    centro_y = img_height // 2

    # Recortar el cuadrante superior derecho
    # Desde (centro_x, 0) hasta (img_width, centro_y)
    cuadrante_superior_derecho = image_bgr[0:centro_y, centro_x:img_width]

    # Guardar el cuadrante como "presidente_1.jpg" en la misma carpeta
    carpeta_imagen = os.path.dirname(img_path)
    zona_path = os.path.join(carpeta_imagen, "presidente_1.jpg")
    cv2.imwrite(zona_path, cuadrante_superior_derecho)

    return f"✅ Procesada: {os.path.basename(img_path)}"


# 📂 Buscar todas las imágenes que contengan "encabezado_" en subcarpetas
img_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if "encabezado_" in file and file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_paths.append(os.path.join(root, file))

total_imgs = len(img_paths)
print(f"📦 Total de imágenes 'encabezado_' detectadas: {total_imgs}")

# 🔧 Calcular número de workers óptimo
max_workers = None
if NUM_WORKERS > 0:
    cpu_count = multiprocessing.cpu_count()
    max_available = max(1, cpu_count - 2)  # Dejar 2 CPUs libres
    max_workers = min(NUM_WORKERS, max_available)
    print(
        f"🚀 Modo paralelo activado: usando {max_workers} workers (CPUs disponibles: {cpu_count})"
    )
else:
    print(f"🐌 Modo secuencial activado")

# 🔁 Procesar imágenes
# Preparar argumentos para cada imagen
args_list = [(idx, img_path, total_imgs) for idx, img_path in enumerate(img_paths, start=1)]

if NUM_WORKERS == 0:
    # Modo secuencial: procesar una por una
    for args in args_list:
        resultado = procesar_imagen(args)
        if "⚠️" in resultado:
            print(resultado)
else:
    # Modo paralelo: usar ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        resultados = executor.map(procesar_imagen, args_list)

        # Mostrar resultados (opcional, para ver errores)
        for resultado in resultados:
            if "⚠️" in resultado:
                print(resultado)

print(f"\n✅ Procesamiento completado: {total_imgs} imágenes")
