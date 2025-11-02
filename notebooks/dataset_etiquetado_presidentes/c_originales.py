# Imports
import os
import random
import shutil

# Prefijo para identificar imágenes (debe aparecer en el nombre del archivo)
PREFIX = "croped_presidente_"

# Variables globales
SEED = 42  # Semilla para reproducibilidad
NUM_MUESTRA = 400  # Número de imágenes a seleccionar aleatoriamente

# Rutas
INPUT_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
OUTPUT_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/data/dataset_etiquetado_presidente/a_originales"


def main():
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Listar todas las imágenes que contengan el PREFIX en todas las subcarpetas
    imagenes = []

    print(f"🔍 Buscando imágenes con prefijo '{PREFIX}' en todas las subcarpetas de:")
    print(f"   {INPUT_PATH}")

    # Recorrer recursivamente todas las subcarpetas
    for root, dirs, files in os.walk(INPUT_PATH):
        for file in files:
            # Verificar si el nombre del archivo contiene el prefijo y es una imagen
            if PREFIX in file and file.lower().endswith((".jpg", ".jpeg", ".png")):
                ruta_completa = os.path.join(root, file)
                imagenes.append(ruta_completa)

    print(f"✓ Total de imágenes encontradas con prefijo '{PREFIX}': {len(imagenes)}")

    # Verificar que hay imágenes disponibles
    if len(imagenes) == 0:
        print(f"❌ ERROR: No se encontraron imágenes con el prefijo '{PREFIX}'")
        return

    # Determinar cuántas imágenes copiar
    if len(imagenes) < NUM_MUESTRA:
        print(
            f"⚠️  ADVERTENCIA: Solo hay {len(imagenes)} imágenes disponibles, pero se solicitaron {NUM_MUESTRA}"
        )
        num_copiar = len(imagenes)
    else:
        num_copiar = NUM_MUESTRA

    # Seleccionar aleatoriamente NUM_MUESTRA imágenes usando el SEED para reproducibilidad
    random.seed(SEED)
    imagenes_seleccionadas = random.sample(imagenes, num_copiar)

    print(f"\n📋 Seleccionadas {num_copiar} imágenes aleatoriamente (SEED={SEED})")
    print(f"📂 Copiando a: {OUTPUT_PATH}\n")

    # Copiar las imágenes seleccionadas con nombres secuenciales
    for i, imagen_origen in enumerate(imagenes_seleccionadas, start=1):
        # Obtener la extensión del archivo original
        extension = os.path.splitext(imagen_origen)[1]

        # Crear nombre secuencial (001.jpg, 002.jpg, etc.)
        nombre_destino = f"{i:03d}{extension}"
        ruta_destino = os.path.join(OUTPUT_PATH, nombre_destino)

        # Copiar el archivo
        shutil.copy2(imagen_origen, ruta_destino)

        if i % 50 == 0:  # Mostrar progreso cada 50 imágenes
            print(f"  📥 Copiadas {i}/{num_copiar} imágenes...")

    print(f"\n✅ Proceso completado exitosamente!")
    print(f"   {num_copiar} imágenes copiadas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
