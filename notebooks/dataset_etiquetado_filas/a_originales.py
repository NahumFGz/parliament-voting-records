# Imports
import os
import random
import shutil

# Prefijo para identificar imágenes
PREFIX = "colyolo_"

# Variables globales
SEED = 42
NUM_MUESTRA = 300

# Rutas
INPUT_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
OUTPUT_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/data/dataset_etiquetado_filas/a_originales"


def main():
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Listar todas las imágenes que contengan el PREFIX
    imagenes = []

    print(f"Buscando imágenes con prefijo '{PREFIX}' en {INPUT_PATH}...")

    # Recorrer todas las subcarpetas
    for root, dirs, files in os.walk(INPUT_PATH):
        for file in files:
            # Verificar si el archivo contiene el prefijo y es una imagen
            if PREFIX in file and file.lower().endswith((".jpg", ".jpeg", ".png")):
                ruta_completa = os.path.join(root, file)
                imagenes.append(ruta_completa)

    print(f"Total de imágenes encontradas: {len(imagenes)}")

    # Verificar que hay suficientes imágenes
    if len(imagenes) < NUM_MUESTRA:
        print(f"ADVERTENCIA: Solo hay {len(imagenes)} imágenes, pero se solicitaron {NUM_MUESTRA}")
        num_copiar = len(imagenes)
    else:
        num_copiar = NUM_MUESTRA

    # Seleccionar aleatoriamente usando el SEED
    random.seed(SEED)
    imagenes_seleccionadas = random.sample(imagenes, num_copiar)

    print(f"Copiando {num_copiar} imágenes seleccionadas aleatoriamente (SEED={SEED})...")

    # Copiar las imágenes con nombres secuenciales
    for i, imagen_origen in enumerate(imagenes_seleccionadas, start=1):
        # Obtener la extensión del archivo original
        extension = os.path.splitext(imagen_origen)[1]

        # Crear nombre secuencial (001.jpg, 002.jpg, etc.)
        nombre_destino = f"{i:03d}{extension}"
        ruta_destino = os.path.join(OUTPUT_PATH, nombre_destino)

        # Copiar el archivo
        shutil.copy2(imagen_origen, ruta_destino)

        if i % 50 == 0:  # Mostrar progreso cada 50 imágenes
            print(f"  Copiadas {i}/{num_copiar} imágenes...")

    print(f"✓ Proceso completado. {num_copiar} imágenes copiadas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
