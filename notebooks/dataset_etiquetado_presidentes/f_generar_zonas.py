import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from ultralytics import YOLO

# ⚙️ Configuraciones para votación
base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
model_path = "/home/nahumfg/GithubProjects/parliament-voting-records/validation/yolo_presidente/experiments/yolo11n_img480_bs32_fold_3/weights/best.pt"

# 🔍 Filtro de nombre de archivo
FILTRO_NOMBRE = "croped_presidente_"  # Solo procesar imágenes que contengan este texto

# 🔧 Configuración de paralelización
# Si es 0, procesa secuencialmente. Si > 0, usa ese número de workers
NUM_WORKERS = 8  # Puedes cambiar este valor según necesites

# 🎯 Configuraciones de márgenes verticales (% del alto de la zona)
# Porcentaje de expansión en el eje Y para la zona de presidente
MARGEN_PRESIDENTE_ARRIBA = 0.01  # 1% hacia arriba
MARGEN_PRESIDENTE_ABAJO = 0.01  # 1% hacia abajo


def aplicar_margenes_verticales(x_min, y_min, x_max, y_max, label, img_height):
    """
    Aplica márgenes verticales a la zona de presidente detectada.

    Args:
        x_min, y_min, x_max, y_max: Coordenadas de la zona
        label: Nombre de la clase (presidente)
        img_height: Alto de la imagen completa

    Returns:
        Tupla (x_min, y_min_ajustada, x_max, y_max_ajustada)
    """
    alto_zona = y_max - y_min
    label_lower = label.lower()

    # Aplicar márgenes para la zona de presidente
    if "presidente" in label_lower:
        # Presidente: expandir arriba y abajo
        incremento_arriba = int(alto_zona * MARGEN_PRESIDENTE_ARRIBA)
        incremento_abajo = int(alto_zona * MARGEN_PRESIDENTE_ABAJO)
        y_min_ajustado = max(0, y_min - incremento_arriba)
        y_max_ajustado = min(y_max + incremento_abajo, img_height)
        return x_min, y_min_ajustado, x_max, y_max_ajustado

    # Si no coincide, devolver original
    return x_min, y_min, x_max, y_max


def procesar_imagen(args):
    """
    Procesa una imagen individual: detecta zonas, aplica márgenes y guarda recortes.

    Args:
        args: Tupla con (idx, image_path, model_path, total_imgs)

    Returns:
        str: Mensaje de estado del procesamiento
    """
    idx, image_path, model_path, total_imgs = args

    # Cargar modelo YOLO en cada proceso
    model = YOLO(model_path)

    if idx % 100 == 0 or idx == 1:
        print(f"\n📊 Progreso: {idx}/{total_imgs} imágenes procesadas")

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        return f"[⚠️] No se pudo leer la imagen: {image_path}"

    # 📍 Predecir zonas
    results = model.predict(
        source=image_bgr, conf=0.01, max_det=1, agnostic_nms=True, verbose=False
    )
    for result in results:
        detecciones = result.boxes
        labels = result.names

        # 📁 Obtener el directorio donde está la imagen
        img_path_obj = Path(image_path)
        img_output_dir = img_path_obj.parent  # Guardar en el mismo directorio

        # Obtener dimensiones de la imagen
        img_height, img_width = image_bgr.shape[:2]

        for i, box in enumerate(detecciones):
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            label = labels[int(box.cls[0])]

            # Aplicar márgenes verticales según el tipo de zona
            x_min, y_min, x_max, y_max = aplicar_margenes_verticales(
                x_min, y_min, x_max, y_max, label, img_height
            )

            # Recortar zona
            zona = image_bgr[y_min:y_max, x_min:x_max]

            # Guardar recorte directamente en el mismo directorio
            zona_path = img_output_dir / f"{label}_{i+1}.jpg"

            cv2.imwrite(str(zona_path), zona)

    return f"✅ Procesada: {img_path_obj.name}"


# 📂 Verificar que el directorio base exista
base_path = Path(base_dir)
if not base_path.exists():
    raise FileNotFoundError(f"El directorio {base_dir} no existe")

# 📋 Buscar imágenes recursivamente en subcarpetas que contengan el filtro
print(f"🔍 Buscando imágenes con '{FILTRO_NOMBRE}' en {base_dir}...")
img_paths = []
for extension in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
    img_paths.extend(base_path.rglob(extension))

# Filtrar solo las que contienen el texto deseado
img_paths = [p for p in img_paths if FILTRO_NOMBRE in p.name]
total_imgs = len(img_paths)
print(f"📦 Total de imágenes con '{FILTRO_NOMBRE}' detectadas: {total_imgs}")

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
args_list = [
    (idx, img_path, model_path, total_imgs) for idx, img_path in enumerate(img_paths, start=1)
]

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
