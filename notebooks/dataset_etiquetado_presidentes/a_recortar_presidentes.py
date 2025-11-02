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
    Procesa una imagen individual: divide en 8 partes (2x4) y guarda las 3 superiores de la derecha.

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

    # Dividir en 2 columnas (izquierda y derecha)
    centro_x = img_width // 2

    # Dividir en 4 filas
    altura_seccion = img_height // 4

    # Carpeta donde guardar
    carpeta_imagen = os.path.dirname(img_path)

    # Recortar las 3 secciones superiores de la columna derecha
    secciones = []
    for i in range(3):
        y_inicio = i * altura_seccion
        y_fin = (i + 1) * altura_seccion

        # Recortar sección derecha (desde centro_x hasta el final)
        seccion = image_bgr[y_inicio:y_fin, centro_x:img_width]
        secciones.append(seccion)

    # Concatenar verticalmente las 3 secciones en una sola imagen
    imagen_concatenada = cv2.vconcat(secciones)

    # Guardar como "presidente_1.jpg"
    zona_path = os.path.join(carpeta_imagen, "croped_presidente_1.jpg")
    cv2.imwrite(zona_path, imagen_concatenada)

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
