import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# ⚙️ Configuraciones
base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
model_path = "/home/nahumfg/GithubProjects/parliament-voting-records/validation/yolo_filas/experiments/yolo11s_img480_bs32_fold_10/weights/best.pt"

# 🏷️ Prefijo para buscar imágenes de entrada y etiquetar imágenes de salida
PREFIX = "colyolo_"  # Busca archivos que empiezan con este prefijo
OUTPUT_PREFIX = "colcolyolo_"  # Prefijo para las imágenes de salida (columnas en columnas)

# 🎯 Configuraciones de márgenes (% del alto/ancho de la zona)
MARGEN_ARRIBA = 0.00  # 2% hacia arriba (0.02 = 2%)
MARGEN_ABAJO = 0.00  # 2% hacia abajo (0.02 = 2%)
MARGEN_IZQUIERDA = 0.00  # 2% hacia la izquierda (0.02 = 2%)
MARGEN_DERECHA = 0.00  # 2% hacia la derecha (0.02 = 2%)

# 🔧 Configuración de paralelización
NUM_WORKERS = 8  #  Si es 0, procesa secuencialmente. Si > 0, usa ese número de workers


# ==================== FUNCIONES DE UTILIDAD ====================


def buscar_todas_imagenes(base_path):
    """
    Busca recursivamente todas las imágenes en el directorio base.

    Args:
        base_path: Path del directorio base

    Returns:
        Lista de Path con todas las imágenes encontradas
    """
    extensiones = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    imagenes = []

    for extension in extensiones:
        imagenes.extend(base_path.rglob(extension))

    return imagenes


def filtrar_imagenes_con_prefix(imagenes, prefix):
    """
    Filtra imágenes que empiezan con el prefijo especificado.

    Args:
        imagenes: Lista de Path con imágenes
        prefix: Prefijo a buscar al inicio del nombre

    Returns:
        Lista de Path con imágenes que empiezan con el prefijo
    """
    return [img for img in imagenes if img.name.startswith(prefix)]


def identificar_carpetas_sin_prefix(todas_imagenes, imagenes_con_prefix):
    """
    Identifica carpetas que tienen imágenes pero no tienen imágenes con el prefijo.

    Args:
        todas_imagenes: Lista de todas las imágenes
        imagenes_con_prefix: Lista de imágenes con el prefijo en el nombre

    Returns:
        Set de Path con carpetas sin imágenes con el prefijo
    """
    carpetas_con_imagenes = set(img.parent for img in todas_imagenes)
    carpetas_con_prefix = set(img.parent for img in imagenes_con_prefix)
    return carpetas_con_imagenes - carpetas_con_prefix


def guardar_reporte_carpetas_sin_prefix(carpetas_sin_prefix, base_dir, prefix):
    """
    Guarda un CSV con las rutas de carpetas sin imágenes con el prefijo.

    Args:
        carpetas_sin_prefix: Set de carpetas sin el prefijo
        base_dir: Directorio base donde guardar el CSV
        prefix: Prefijo buscado
    """
    if not carpetas_sin_prefix:
        print(f"✅ Todas las carpetas con imágenes tienen al menos una imagen con '{prefix}'")
        return

    print(f"📋 Se encontraron {len(carpetas_sin_prefix)} carpetas sin imágenes con '{prefix}'")

    csv_path = os.path.join(base_dir, f"carpetas_sin_{prefix.rstrip('_')}.csv")
    df_sin_prefix = pd.DataFrame(
        {"ruta_carpeta": sorted([str(carpeta) for carpeta in carpetas_sin_prefix])}
    )
    df_sin_prefix.to_csv(csv_path, index=False)
    print(f"💾 CSV guardado en: {csv_path}")


def configurar_workers(num_workers):
    """
    Configura el número óptimo de workers para procesamiento paralelo.

    Args:
        num_workers: Número de workers solicitado (0 = secuencial)

    Returns:
        Número de workers a usar (None si modo secuencial)
    """
    if num_workers == 0:
        print("🐌 Modo secuencial activado")
        return None

    cpu_count = multiprocessing.cpu_count()
    max_available = max(1, cpu_count - 2)  # Dejar 2 CPUs libres
    max_workers = min(num_workers, max_available)
    print(
        f"🚀 Modo paralelo activado: usando {max_workers} workers (CPUs disponibles: {cpu_count})"
    )
    return max_workers


def ejecutar_procesamiento(args_list, max_workers):
    """
    Ejecuta el procesamiento de imágenes en modo paralelo o secuencial.

    Args:
        args_list: Lista de argumentos para procesar_imagen
        max_workers: Número de workers (None = modo secuencial)
    """
    if max_workers is None:
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


# ==================== FUNCIONES DE PROCESAMIENTO ====================


def aplicar_margenes(x_min, y_min, x_max, y_max, img_height, img_width):
    """
    Aplica márgenes horizontales y verticales a una zona detectada.

    Args:
        x_min, y_min, x_max, y_max: Coordenadas de la zona
        img_height: Alto de la imagen completa
        img_width: Ancho de la imagen completa

    Returns:
        Tupla (x_min_ajustado, y_min_ajustado, x_max_ajustado, y_max_ajustado)
    """
    alto_zona = y_max - y_min
    ancho_zona = x_max - x_min

    # Calcular incrementos verticales
    incremento_arriba = int(alto_zona * MARGEN_ARRIBA)
    incremento_abajo = int(alto_zona * MARGEN_ABAJO)

    # Calcular incrementos horizontales
    incremento_izquierda = int(ancho_zona * MARGEN_IZQUIERDA)
    incremento_derecha = int(ancho_zona * MARGEN_DERECHA)

    # Aplicar márgenes asegurando que no salgan de los límites de la imagen
    x_min_ajustado = max(0, x_min - incremento_izquierda)
    x_max_ajustado = min(x_max + incremento_derecha, img_width)
    y_min_ajustado = max(0, y_min - incremento_arriba)
    y_max_ajustado = min(y_max + incremento_abajo, img_height)

    return x_min_ajustado, y_min_ajustado, x_max_ajustado, y_max_ajustado


def procesar_imagen(args):
    """
    Procesa una imagen individual: detecta zonas y guarda recortes en carpeta propia.

    Args:
        args: Tupla con (idx, img_path, model_path, output_prefix, total_imgs)

    Returns:
        str: Mensaje de estado del procesamiento
    """
    idx, img_path, model_path, output_prefix, total_imgs = args

    # Cargar modelo YOLO en cada proceso
    model = YOLO(model_path)

    if idx % 100 == 0 or idx == 1:
        print(f"\n📊 Progreso: {idx}/{total_imgs} imágenes procesadas")

    image_bgr = cv2.imread(str(img_path))

    if image_bgr is None:
        return f"[⚠️] No se pudo leer la imagen: {img_path}"

    # Obtener dimensiones de la imagen
    img_height, img_width = image_bgr.shape[:2]

    # Crear carpeta de salida con el nombre del archivo (sin extensión)
    img_dir = img_path.parent
    nombre_sin_extension = img_path.stem  # nombre sin .jpg
    carpeta_salida = img_dir / nombre_sin_extension

    # Crear la carpeta si no existe
    carpeta_salida.mkdir(exist_ok=True)

    # 📍 Predecir zonas
    results = model.predict(
        source=image_bgr, conf=0.001, iou=0.7, max_det=3, agnostic_nms=True, verbose=False
    )
    for result in results:
        detecciones = result.boxes
        labels = result.names

        for i, box in enumerate(detecciones):
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            label = labels[int(box.cls[0])]

            # Aplicar márgenes horizontales y verticales
            x_min, y_min, x_max, y_max = aplicar_margenes(
                x_min, y_min, x_max, y_max, img_height, img_width
            )

            # Recortar zona
            zona = image_bgr[y_min:y_max, x_min:x_max]

            # Guardar recorte con prefijo en la carpeta creada
            zona_path = carpeta_salida / f"{output_prefix}{label}_{i+1}.jpg"

            cv2.imwrite(str(zona_path), zona)

    return f"✅ Procesada: {img_path.name}"


# ==================== PROGRAMA PRINCIPAL ====================

if __name__ == "__main__":
    # 📋 Paso 1: Buscar todas las imágenes
    print("🔍 Buscando imágenes...")
    base_path = Path(base_dir)
    todas_imagenes = buscar_todas_imagenes(base_path)
    print(f"   Encontradas {len(todas_imagenes)} imágenes en total")

    # 📋 Paso 2: Filtrar imágenes con el PREFIX
    imagenes_con_prefix = filtrar_imagenes_con_prefix(todas_imagenes, PREFIX)
    total_imgs = len(imagenes_con_prefix)
    print(f"📦 Total de imágenes con prefijo '{PREFIX}': {total_imgs}")

    # 📊 Paso 3: Identificar y reportar carpetas sin prefix
    carpetas_sin_prefix = identificar_carpetas_sin_prefix(todas_imagenes, imagenes_con_prefix)
    guardar_reporte_carpetas_sin_prefix(carpetas_sin_prefix, base_dir, PREFIX)

    # ⚠️ Validar que hay imágenes para procesar
    if total_imgs == 0:
        print(f"⚠️ No se encontraron imágenes con prefijo '{PREFIX}'")
        exit()

    # 🔧 Paso 4: Configurar workers
    max_workers = configurar_workers(NUM_WORKERS)

    # 🔁 Paso 5: Preparar argumentos para procesamiento
    args_list = [
        (idx, img_path, model_path, OUTPUT_PREFIX, total_imgs)
        for idx, img_path in enumerate(imagenes_con_prefix, start=1)
    ]

    # 🚀 Paso 6: Ejecutar procesamiento
    print(f"\n▶️  Iniciando procesamiento de {total_imgs} imágenes...")
    ejecutar_procesamiento(args_list, max_workers)

    # ✅ Finalizar
    print(f"\n✅ Procesamiento completado: {total_imgs} imágenes")
