#!/usr/bin/env python3
"""
Script para listar todas las imágenes .jpg que contienen la palabra 'columnas'
en la carpeta data/procesamiento_todas_votaciones/b_zonas
"""

import os
import random
import shutil
from pathlib import Path

# Variables globales
SEED = 42
NUM_MUESTRA = 300

# Rutas
INPUT_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
OUTPUT_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/data/dataset_etiquetado_columnas/a_originales"


def encontrar_imagenes_columnas(directorio_base, palabra_clave="columnas"):
    """
    Encuentra todas las imágenes .jpg que contienen la palabra clave en su nombre.

    Args:
        directorio_base: Directorio donde buscar
        palabra_clave: Palabra a buscar en el nombre del archivo (default: "columnas")

    Returns:
        Lista de rutas completas de las imágenes encontradas
    """
    imagenes_columnas = []

    # Recorrer recursivamente todos los archivos
    for root, dirs, files in os.walk(directorio_base):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                if palabra_clave.lower() in file.lower():
                    ruta_completa = os.path.join(root, file)
                    imagenes_columnas.append(ruta_completa)

    return imagenes_columnas


def copiar_imagen_con_nuevo_nombre(ruta_imagen, carpeta_destino):
    """
    Copia una imagen a la carpeta destino con un nuevo nombre basado en la carpeta padre.

    Args:
        ruta_imagen: Ruta completa de la imagen original
        carpeta_destino: Carpeta donde guardar la imagen

    Returns:
        Nueva ruta de la imagen copiada
    """
    # Obtener el nombre de la carpeta padre
    ruta_obj = Path(ruta_imagen)
    carpeta_padre = ruta_obj.parent.name

    # Crear el nuevo nombre: nombre_carpeta_columnas_.jpg
    nuevo_nombre = carpeta_padre + "columnas_.jpg"

    # Crear la carpeta de destino si no existe
    Path(carpeta_destino).mkdir(parents=True, exist_ok=True)

    # Ruta completa del archivo de destino
    ruta_destino = Path(carpeta_destino) / nuevo_nombre

    # Copiar la imagen
    shutil.copy2(ruta_imagen, ruta_destino)

    return str(ruta_destino)


def main():
    # Configurar seed para reproducibilidad
    random.seed(SEED)

    print(f"Buscando imágenes con la palabra 'columnas' en: {INPUT_PATH}")
    print(f"Seed: {SEED}, Muestra: {NUM_MUESTRA} imágenes")
    print("-" * 80)

    # Encontrar todas las imágenes
    imagenes_todas = encontrar_imagenes_columnas(INPUT_PATH)
    print(f"\nTotal de imágenes encontradas: {len(imagenes_todas)}")

    # Tomar una muestra aleatoria
    if len(imagenes_todas) > NUM_MUESTRA:
        imagenes = random.sample(imagenes_todas, NUM_MUESTRA)
        print(f"Muestra aleatoria seleccionada: {len(imagenes)} imágenes")
    else:
        imagenes = imagenes_todas
        print(f"Total de imágenes menor a la muestra, usando todas: {len(imagenes)}")

    print("-" * 80)

    # Copiar las imágenes a la carpeta de destino
    print(f"\nCopiando imágenes a: {OUTPUT_PATH}")
    print("-" * 80)

    for i, ruta_imagen in enumerate(imagenes, 1):
        nueva_ruta = copiar_imagen_con_nuevo_nombre(ruta_imagen, OUTPUT_PATH)
        if i <= 10:  # Mostrar las primeras 10
            print(f"{i}. {Path(ruta_imagen).name} -> {Path(nueva_ruta).name}")

    print(f"\n✓ {len(imagenes)} imágenes copiadas exitosamente")
    print(f"  Directorio: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
