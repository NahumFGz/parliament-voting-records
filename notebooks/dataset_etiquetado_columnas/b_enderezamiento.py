import os
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from scipy import ndimage


def detectar_angulo_proyeccion(gray, rango_angulos=(-5, 5), paso=0.1):
    """
    Detecta el ángulo de inclinación usando proyección de varianza.
    Muy preciso para documentos con texto.
    """
    # Binarizar
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary  # Invertir (texto negro sobre blanco)

    # Buscar en rango con pasos finos
    angles = np.arange(rango_angulos[0], rango_angulos[1], paso)
    scores = []

    for angle in angles:
        # Rotar imagen
        rotated = ndimage.rotate(binary, angle, reshape=False, order=0)

        # Calcular proyección horizontal (suma de píxeles por fila)
        projection = np.sum(rotated, axis=1)

        # La varianza será máxima cuando las líneas estén alineadas
        score = np.var(projection)
        scores.append(score)

    # El ángulo con mayor varianza es el correcto
    best_idx = np.argmax(scores)
    best_angle = angles[best_idx]

    return best_angle


def procesar_imagen(args):
    """
    Procesa una sola imagen (función auxiliar para paralelización).

    Parámetros:
    - args: Tupla con (file_name, input_path, output_path, rango_angulos, paso)

    Retorna:
    - Tupla: (file_name, angle, success, error_msg)
    """
    file_name, input_path, output_path, rango_angulos, paso = args

    input_file = os.path.join(input_path, file_name)
    output_file = os.path.join(output_path, file_name)

    try:
        # Cargar imagen
        img = cv2.imread(input_file)
        if img is None:
            return (file_name, 0, False, "No se pudo cargar la imagen")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectar ángulo
        angle = detectar_angulo_proyeccion(gray, rango_angulos, paso)

        # Rotar imagen
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        # Guardar
        cv2.imwrite(output_file, rotated)

        return (file_name, angle, True, None)

    except Exception as e:
        return (file_name, 0, False, str(e))


def enderezar_carpeta(input_path, output_path, rango_angulos=(-5, 5), paso=0.1, n_workers=None):
    """
    Endereza todas las imágenes de una carpeta con procesamiento paralelo.

    Parámetros:
    - input_path: Carpeta con imágenes comprimidas
    - output_path: Carpeta para guardar enderezadas
    - rango_angulos: Rango de búsqueda de ángulos (-5, 5 recomendado)
    - paso: Precisión de búsqueda (0.1° recomendado)
    - n_workers: Número de hilos del procesador (None = todos disponibles, 1 = secuencial)
    """
    # Crear carpeta de salida
    os.makedirs(output_path, exist_ok=True)

    # Obtener lista de archivos
    archivos = [
        f
        for f in os.listdir(input_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
    ]

    if not archivos:
        print("⚠ No se encontraron imágenes en la carpeta de entrada")
        return

    # Determinar número de workers
    cpus_disponibles = cpu_count()
    if n_workers is None:
        n_workers = cpus_disponibles
    else:
        n_workers = min(n_workers, cpus_disponibles)

    print(f"{'='*60}")
    print(f"Enderezando documentos...")
    print(f"  CPUs disponibles: {cpus_disponibles}")
    print(f"  Workers a usar: {n_workers}")
    print(f"  Archivos a procesar: {len(archivos)}")
    print(f"{'='*60}\n")

    # Preparar argumentos para cada archivo
    args_list = [(f, input_path, output_path, rango_angulos, paso) for f in archivos]

    archivos_procesados = 0
    archivos_fallidos = 0
    angulos_detectados = []

    # Procesar en paralelo o secuencial
    if n_workers == 1:
        # Procesamiento secuencial
        resultados = [procesar_imagen(args) for args in args_list]
    else:
        # Procesamiento paralelo
        with Pool(processes=n_workers) as pool:
            resultados = pool.map(procesar_imagen, args_list)

    # Procesar resultados
    for file_name, angle, success, error_msg in resultados:
        if success:
            print(f"✓ {file_name} → Ángulo: {angle:.3f}°")
            archivos_procesados += 1
            angulos_detectados.append(abs(angle))
        else:
            print(f"✗ {file_name} → Error: {error_msg}")
            archivos_fallidos += 1

    # Resumen
    print(f"\n{'='*60}")
    print(f"✅ Enderezado completado")
    print(f"   Archivos procesados: {archivos_procesados}")
    print(f"   Archivos fallidos: {archivos_fallidos}")
    if angulos_detectados:
        print(f"   Ángulo promedio: {np.mean(angulos_detectados):.3f}°")
        print(f"   Ángulo máximo: {np.max(angulos_detectados):.3f}°")
        print(f"   Ángulo mínimo: {np.min(angulos_detectados):.3f}°")
    print(f"{'='*60}")


# ============================================
# EJECUTAR
# ============================================

if __name__ == "__main__":
    input_path = "../../data/dataset_etiquetado_columnas/a_originales"
    output_path = "../../data/dataset_etiquetado_columnas/b_enderezados"

    enderezar_carpeta(
        input_path=input_path,
        output_path=output_path,
        rango_angulos=(-5, 5),  # Busca inclinaciones entre -5° y +5°
        paso=0.1,  # Precisión de 0.1 grados
        n_workers=13,  # None = usar todos los CPUs disponibles
        # 1 = secuencial (sin paralelización)
        # 4 = usar 4 hilos
        # 8 = usar 8 hilos
    )

    print("\n✅ Proceso completado. Revisa la carpeta:", output_path)
