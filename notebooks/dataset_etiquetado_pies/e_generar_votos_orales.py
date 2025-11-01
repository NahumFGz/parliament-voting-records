import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ⚙️ Configuraciones
base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
model_path = "/home/nahumfg/GithubProjects/parliament-voting-records/validation/yolo_pies/experiments/yolo11n_img320_bs32_fold_3/weights/best.pt"

# 🏷️ Prefijo para las imágenes generadas
PREFIX_ORIGINAL = "pie_"
PREFIX_CROPED = "pieyolo_"

# 🎯 Configuración de margen
MARGEN_ABAJO = 0.02  # 2% de reducción en la parte baja de las regiones

# 🔧 Configuración de paralelización
NUM_WORKERS = 8  #  Si es 0, procesa secuencialmente. Si > 0, usa ese número de workers


# ==================== FUNCIONES DE UTILIDAD ====================


def buscar_todas_imagenes(base_path):
    """
    Busca imágenes solo en el primer nivel de subcarpetas dentro del directorio base.
    No busca recursivamente en niveles más profundos.

    Estructura de búsqueda:
    - base_dir/carpeta_x/imagen.jpg ✓
    - base_dir/carpeta_x/carpeta_y/imagen.jpg ✗

    Args:
        base_path: Path del directorio base

    Returns:
        Lista de Path con todas las imágenes encontradas en el primer nivel
    """
    extensiones = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    imagenes = []

    # Buscar solo en el primer nivel de subcarpetas (profundidad 1)
    for extension in extensiones:
        imagenes.extend(base_path.glob(f"*/{extension}"))

    return imagenes


def filtrar_imagenes_con_prefijo(imagenes, prefijo):
    """
    Filtra imágenes que contienen el prefijo especificado en el nombre.

    Args:
        imagenes: Lista de Path con imágenes
        prefijo: Prefijo a buscar en el nombre de archivo

    Returns:
        Lista de Path con imágenes que contienen el prefijo en el nombre
    """
    return [img for img in imagenes if prefijo in img.name]


def identificar_carpetas_sin_prefijo(todas_imagenes, imagenes_con_prefijo):
    """
    Identifica carpetas que tienen imágenes pero no tienen imágenes con el prefijo.

    Args:
        todas_imagenes: Lista de todas las imágenes
        imagenes_con_prefijo: Lista de imágenes con el prefijo en el nombre

    Returns:
        Set de Path con carpetas sin imágenes con el prefijo
    """
    carpetas_con_imagenes = set(img.parent for img in todas_imagenes)
    carpetas_con_prefijo = set(img.parent for img in imagenes_con_prefijo)
    return carpetas_con_imagenes - carpetas_con_prefijo


def guardar_reporte_carpetas_sin_prefijo(carpetas_sin_prefijo, base_dir, prefijo):
    """
    Guarda un CSV con las rutas de carpetas sin imágenes con el prefijo.

    Args:
        carpetas_sin_prefijo: Set de carpetas sin el prefijo
        base_dir: Directorio base donde guardar el CSV
        prefijo: Prefijo que se buscaba
    """
    if not carpetas_sin_prefijo:
        print(f"✅ Todas las carpetas con imágenes tienen al menos una imagen con '{prefijo}'")
        return

    print(f"📋 Se encontraron {len(carpetas_sin_prefijo)} carpetas sin imágenes con '{prefijo}'")

    csv_path = os.path.join(base_dir, f"carpetas_sin_{prefijo.replace('_', '')}.csv")
    df_sin_prefijo = pd.DataFrame(
        {"ruta_carpeta": sorted([str(carpeta) for carpeta in carpetas_sin_prefijo])}
    )
    df_sin_prefijo.to_csv(csv_path, index=False)
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


def procesar_imagen(args):
    """
    Procesa una imagen individual: detecta zonas, las extiende a los bordes y extrae el voto oral.

    - resultado: se extiende hasta el borde izquierdo y superior
    - grupo_parlamentario: se extiende hasta arriba, izquierda y derecha
    - voto_oral: área de la imagen que NO está cubierta por las regiones anteriores

    Args:
        args: Tupla con (idx, img_path, model_path, prefix_croped, total_imgs)

    Returns:
        str: Mensaje de estado del procesamiento
    """
    idx, img_path, model_path, prefix_croped, total_imgs = args

    # Cargar modelo YOLO en cada proceso
    model = YOLO(model_path)

    if idx % 100 == 0 or idx == 1:
        print(f"\n📊 Progreso: {idx}/{total_imgs} imágenes procesadas")

    image_bgr = cv2.imread(str(img_path))

    if image_bgr is None:
        return f"[⚠️] No se pudo leer la imagen: {img_path}"

    # Obtener dimensiones de la imagen
    img_height, img_width = image_bgr.shape[:2]

    # 📍 Predecir zonas
    results = model.predict(
        source=image_bgr,
        conf=0.001,  # Mismo que en entrenamiento
        iou=0.7,  # Mismo que en entrenamiento
        max_det=2,  # Máximo 2 detecciones
        agnostic_nms=True,  # NMS agnóstico entre clases
        verbose=False,
    )

    # Obtener directorio donde está la imagen original
    img_dir = str(img_path.parent)

    # Diccionario para almacenar las regiones detectadas
    regiones = {}

    for result in results:
        detecciones = result.boxes
        labels = result.names

        for i, box in enumerate(detecciones):
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            label = labels[int(box.cls[0])]

            # Guardar coordenadas originales por tipo de región
            if label not in regiones:
                regiones[label] = []
            regiones[label].append((x_min, y_min, x_max, y_max))

    # Calcular la altura mínima entre las 2 regiones para el recorte final
    altura_minima = img_height  # Valor por defecto

    if "resultado" in regiones:
        for x_min, y_min, x_max, y_max in regiones["resultado"]:
            alto_region = y_max - y_min
            altura_minima = min(altura_minima, alto_region)

    if "grupo_parlamentario" in regiones:
        for x_min, y_min, x_max, y_max in regiones["grupo_parlamentario"]:
            alto_region = y_max - y_min
            altura_minima = min(altura_minima, alto_region)

    # Crear máscara para marcar las regiones ocupadas
    mascara_ocupada = np.zeros((img_height, img_width), dtype=np.uint8)

    # Procesar región "resultado" - extender hasta borde izquierdo y superior
    if "resultado" in regiones:
        for x_min, y_min, x_max, y_max in regiones["resultado"]:
            # Extender hasta el borde izquierdo (x_min = 0) y superior (y_min = 0)
            x_min_ext = 0
            y_min_ext = 0
            x_max_ext = x_max

            # Aplicar margen inferior: reducir 2% de la altura
            alto_region = y_max - y_min
            reduccion_abajo = int(alto_region * MARGEN_ABAJO)
            y_max_ext = y_max - reduccion_abajo

            # Marcar región en la máscara
            mascara_ocupada[y_min_ext:y_max_ext, x_min_ext:x_max_ext] = 255

    # Procesar región "grupo_parlamentario" - extender hasta arriba, izquierda y derecha
    if "grupo_parlamentario" in regiones:
        for x_min, y_min, x_max, y_max in regiones["grupo_parlamentario"]:
            # Extender hasta arriba (y_min = 0), izquierda (x_min = 0) y derecha (x_max = img_width)
            x_min_ext = 0
            y_min_ext = 0
            x_max_ext = img_width

            # Aplicar margen inferior: reducir 2% de la altura
            alto_region = y_max - y_min
            reduccion_abajo = int(alto_region * MARGEN_ABAJO)
            y_max_ext = y_max - reduccion_abajo

            # Marcar región en la máscara
            mascara_ocupada[y_min_ext:y_max_ext, x_min_ext:x_max_ext] = 255

    # Invertir la máscara para obtener la región NO ocupada (voto_oral)
    mascara_voto_oral = cv2.bitwise_not(mascara_ocupada)

    # Extraer la región de voto_oral usando la máscara
    voto_oral = cv2.bitwise_and(image_bgr, image_bgr, mask=mascara_voto_oral)

    # Recortar la imagen desde la altura mínima hacia abajo
    # Esto elimina la parte superior que ya está cubierta por las regiones extendidas
    voto_oral_recortado = voto_oral[altura_minima:img_height, 0:img_width]

    # Guardar la región de voto_oral recortada
    voto_oral_path = os.path.join(img_dir, f"{prefix_croped}voto_oral.jpg")
    cv2.imwrite(voto_oral_path, voto_oral_recortado)

    return f"✅ Procesada: {img_path.name}"


# ==================== PROGRAMA PRINCIPAL ====================

if __name__ == "__main__":
    # 📋 Paso 1: Buscar todas las imágenes
    print("🔍 Buscando imágenes...")
    base_path = Path(base_dir)
    todas_imagenes = buscar_todas_imagenes(base_path)
    print(f"   Encontradas {len(todas_imagenes)} imágenes en total")

    # 📋 Paso 2: Filtrar imágenes con PREFIX_ORIGINAL
    imagenes_con_prefijo = filtrar_imagenes_con_prefijo(todas_imagenes, PREFIX_ORIGINAL)
    total_imgs = len(imagenes_con_prefijo)
    print(f"📦 Total de imágenes con '{PREFIX_ORIGINAL}' en el nombre: {total_imgs}")

    # 📊 Paso 3: Identificar y reportar carpetas sin el prefijo
    carpetas_sin_prefijo = identificar_carpetas_sin_prefijo(todas_imagenes, imagenes_con_prefijo)
    guardar_reporte_carpetas_sin_prefijo(carpetas_sin_prefijo, base_dir, PREFIX_ORIGINAL)

    # ⚠️ Validar que hay imágenes para procesar
    if total_imgs == 0:
        print(f"⚠️ No se encontraron imágenes con '{PREFIX_ORIGINAL}' en el nombre")
        exit()

    # 🔧 Paso 4: Configurar workers
    max_workers = configurar_workers(NUM_WORKERS)

    # 🔁 Paso 5: Preparar argumentos para procesamiento
    args_list = [
        (idx, img_path, model_path, PREFIX_CROPED, total_imgs)
        for idx, img_path in enumerate(imagenes_con_prefijo, start=1)
    ]

    # 🚀 Paso 6: Ejecutar procesamiento
    print(f"\n▶️  Iniciando procesamiento de {total_imgs} imágenes...")
    ejecutar_procesamiento(args_list, max_workers)

    # ✅ Finalizar
    print(f"\n✅ Procesamiento completado: {total_imgs} imágenes")
