import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from ultralytics import YOLO

# ⚙️ Configuraciones para votación
input_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/a_originales"
output_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
model_path = "/home/nahumfg/GithubProjects/parliament-voting-records/validation/yolo_zonas/experiments/yolo11n_img480_bs32_fold_9/weights/best.pt"

# 🔧 Configuración de paralelización
# Si es 0, procesa secuencialmente. Si > 0, usa ese número de workers
NUM_WORKERS = 8  # Puedes cambiar este valor según necesites

# 🎯 Configuraciones de márgenes verticales (% del alto de la zona)
# Porcentaje de expansión en el eje Y para cada tipo de zona
MARGEN_ENCABEZADO_ABAJO = 0.02  # 5% hacia abajo (0.05 = 5%)
MARGEN_COLUMNAS_ARRIBA = 0.02  # 5% hacia arriba
MARGEN_COLUMNAS_ABAJO = 0.02  # 5% hacia abajo
MARGEN_PIE_ARRIBA = 0.02  # 5% hacia arriba


def aplicar_margenes_verticales(x_min, y_min, x_max, y_max, label, img_height):
    """
    Aplica márgenes verticales según el tipo de zona detectada.

    Args:
        x_min, y_min, x_max, y_max: Coordenadas de la zona
        label: Nombre de la clase (encabezado, columnas, pie)
        img_height: Alto de la imagen completa

    Returns:
        Tupla (x_min, y_min_ajustada, x_max, y_max_ajustada)
    """
    alto_zona = y_max - y_min
    label_lower = label.lower()

    # Aplicar márgenes según el tipo de zona
    if "encabezado" in label_lower:
        # Encabezado: expandir hacia abajo
        incremento = int(alto_zona * MARGEN_ENCABEZADO_ABAJO)
        y_max_ajustado = min(y_max + incremento, img_height)
        return x_min, y_min, x_max, y_max_ajustado

    elif "columnas" in label_lower:
        # Columnas: expandir arriba y abajo
        incremento_arriba = int(alto_zona * MARGEN_COLUMNAS_ARRIBA)
        incremento_abajo = int(alto_zona * MARGEN_COLUMNAS_ABAJO)
        y_min_ajustado = max(0, y_min - incremento_arriba)
        y_max_ajustado = min(y_max + incremento_abajo, img_height)
        return x_min, y_min_ajustado, x_max, y_max_ajustado

    elif "pie" in label_lower:
        # Pie: expandir hacia arriba
        incremento = int(alto_zona * MARGEN_PIE_ARRIBA)
        y_min_ajustado = max(0, y_min - incremento)
        return x_min, y_min_ajustado, x_max, y_max

    # Si no coincide con ninguna zona conocida, devolver original
    return x_min, y_min, x_max, y_max


def procesar_imagen(args):
    """
    Procesa una imagen individual: detecta zonas, aplica márgenes y guarda recortes.

    Args:
        args: Tupla con (idx, img_file, input_dir, output_dir, model_path, total_imgs)

    Returns:
        str: Mensaje de estado del procesamiento
    """
    idx, img_file, input_dir, output_dir, model_path, total_imgs = args

    # Cargar modelo YOLO en cada proceso
    model = YOLO(model_path)

    if idx % 100 == 0 or idx == 1:
        print(f"\n📊 Progreso: {idx}/{total_imgs} imágenes procesadas")

    image_path = os.path.join(input_dir, img_file)
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        return f"[⚠️] No se pudo leer la imagen: {image_path}"

    # 📍 Predecir zonas
    results = model.predict(
        source=image_bgr, conf=0.25, max_det=3, agnostic_nms=True, verbose=False
    )
    for result in results:
        detecciones = result.boxes
        labels = result.names

        # 📁 Crear carpeta con nombre base de la imagen
        base_name = os.path.splitext(img_file)[0]
        img_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # 💾 Guardar imagen original
        original_path = os.path.join(img_output_dir, "original.jpg")
        cv2.imwrite(original_path, image_bgr)

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

            # Guardar recorte
            zona_path = os.path.join(img_output_dir, f"{label}_{i+1}.jpg")

            cv2.imwrite(zona_path, zona)

    return f"✅ Procesada: {img_file}"


# 📂 Crear carpeta destino si no existe
os.makedirs(output_dir, exist_ok=True)

# 📋 Listar imágenes válidas
img_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
total_imgs = len(img_files)
print(f"📦 Total de imágenes detectadas: {total_imgs}")

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
    (idx, img_file, input_dir, output_dir, model_path, total_imgs)
    for idx, img_file in enumerate(img_files, start=1)
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
