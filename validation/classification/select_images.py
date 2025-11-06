#!/usr/bin/env python3
"""
Script para seleccionar 300 imágenes aleatorias de cada clase (asistencia, otros, votacion)
manteniendo una distribución balanceada por períodos presidenciales.

Autores: Dataset para entrenamiento de clasificación
Fecha: 2024
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Configuración de rutas
SOURCE_BASE = (
    "/home/nahumfg/GithubProjects/parliament-voting-records/data/dataset_classification/all"
)
DEST_BASE = (
    "/home/nahumfg/GithubProjects/parliament-voting-records/data/dataset_classification/selected"
)

# Clases disponibles
CLASSES = ["asistencia", "otros", "votacion"]

# Número de imágenes a seleccionar por clase
IMAGES_PER_CLASS = 500

# Semilla para reproducibilidad
RANDOM_SEED = 42


def extract_period_from_filename(filename: str) -> str:
    """
    Extrae el período presidencial del nombre del archivo.

    Args:
        filename: Nombre del archivo en formato XXX_ppYYYY_YYYY_paYYYY_YYYY_legX_page_Y.png

    Returns:
        Período en formato YYYY_YYYY
    """
    pattern = r"_pp(\d{4}_\d{4})_"
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return "unknown"


def get_images_by_period(class_path: str) -> Dict[str, List[str]]:
    """
    Organiza las imágenes de una clase por período presidencial.

    Args:
        class_path: Ruta a la carpeta de la clase

    Returns:
        Diccionario con período como clave y lista de archivos como valor
    """
    images_by_period = defaultdict(list)

    if not os.path.exists(class_path):
        print(f"Advertencia: La ruta {class_path} no existe")
        return images_by_period

    for filename in os.listdir(class_path):
        if filename.endswith(".png"):
            period = extract_period_from_filename(filename)
            images_by_period[period].append(filename)

    return images_by_period


def calculate_proportional_selection(
    images_by_period: Dict[str, List[str]], target_count: int
) -> Dict[str, int]:
    """
    Calcula cuántas imágenes seleccionar de cada período de manera proporcional.

    Args:
        images_by_period: Diccionario con imágenes organizadas por período
        target_count: Número total de imágenes a seleccionar

    Returns:
        Diccionario con la cantidad de imágenes a seleccionar por período
    """
    total_images = sum(len(images) for images in images_by_period.values())
    selection_count = {}

    if total_images == 0:
        return selection_count

    # Calcular selección proporcional
    remaining_target = target_count
    periods = list(images_by_period.keys())

    for i, period in enumerate(periods):
        available = len(images_by_period[period])

        if i == len(periods) - 1:  # Último período, asignar lo que quede
            selection_count[period] = min(available, remaining_target)
        else:
            proportion = available / total_images
            count = min(available, int(proportion * target_count))
            selection_count[period] = count
            remaining_target -= count

    return selection_count


def select_images_from_class(class_name: str, target_count: int = IMAGES_PER_CLASS) -> List[str]:
    """
    Selecciona imágenes aleatorias de una clase manteniendo distribución balanceada.

    Args:
        class_name: Nombre de la clase (asistencia, otros, votacion)
        target_count: Número de imágenes a seleccionar

    Returns:
        Lista de rutas completas de archivos seleccionados
    """
    class_path = os.path.join(SOURCE_BASE, class_name)
    images_by_period = get_images_by_period(class_path)

    if not images_by_period:
        print(f"No se encontraron imágenes para la clase {class_name}")
        return []

    # Mostrar distribución disponible
    print(f"\nClase '{class_name}' - Distribución disponible:")
    total_available = 0
    for period, images in images_by_period.items():
        print(f"  {period}: {len(images)} imágenes")
        total_available += len(images)
    print(f"  Total: {total_available} imágenes")

    # Si hay menos imágenes disponibles que las solicitadas, ajustar
    if total_available < target_count:
        print(
            f"Advertencia: Solo hay {total_available} imágenes disponibles, menos que las {target_count} solicitadas"
        )
        target_count = total_available

    # Calcular selección proporcional
    selection_count = calculate_proportional_selection(images_by_period, target_count)

    print(f"Selección planificada:")
    selected_files = []

    for period, count in selection_count.items():
        if count > 0:
            available_images = images_by_period[period]
            selected = random.sample(available_images, count)

            # Agregar ruta completa
            for img in selected:
                full_path = os.path.join(class_path, img)
                selected_files.append(full_path)

            print(f"  {period}: {count} imágenes seleccionadas")

    print(f"Total seleccionadas: {len(selected_files)} imágenes")
    return selected_files


def create_destination_structure():
    """Crea la estructura de directorios de destino."""
    for class_name in CLASSES:
        dest_dir = os.path.join(DEST_BASE, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Directorio creado/verificado: {dest_dir}")


def copy_selected_images(selected_images: Dict[str, List[str]]):
    """
    Copia las imágenes seleccionadas al directorio de destino.

    Args:
        selected_images: Diccionario con clase como clave y lista de rutas como valor
    """
    print("\n" + "=" * 60)
    print("INICIANDO COPIA DE IMÁGENES")
    print("=" * 60)

    total_copied = 0

    for class_name, image_paths in selected_images.items():
        dest_dir = os.path.join(DEST_BASE, class_name)
        class_copied = 0

        print(f"\nCopiando imágenes de la clase '{class_name}'...")

        for src_path in image_paths:
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                dest_path = os.path.join(dest_dir, filename)

                try:
                    shutil.copy2(src_path, dest_path)
                    class_copied += 1
                    total_copied += 1

                    if class_copied % 50 == 0:  # Mostrar progreso cada 50 archivos
                        print(f"  Copiadas {class_copied} imágenes...")

                except Exception as e:
                    print(f"Error copiando {src_path}: {e}")
            else:
                print(f"Advertencia: Archivo no encontrado: {src_path}")

        print(f"  ✓ Clase '{class_name}': {class_copied} imágenes copiadas")

    print(f"\n🎉 PROCESO COMPLETADO: {total_copied} imágenes copiadas en total")


def show_final_summary():
    """Muestra un resumen final del dataset creado."""
    print("\n" + "=" * 60)
    print("RESUMEN FINAL DEL DATASET DE ENTRENAMIENTO")
    print("=" * 60)

    total_images = 0
    for class_name in CLASSES:
        dest_dir = os.path.join(DEST_BASE, class_name)
        if os.path.exists(dest_dir):
            count = len([f for f in os.listdir(dest_dir) if f.endswith(".png")])
            print(f"Clase '{class_name}': {count} imágenes")
            total_images += count
        else:
            print(f"Clase '{class_name}': 0 imágenes (directorio no existe)")

    print(f"\nTOTAL: {total_images} imágenes en el dataset de entrenamiento")
    print(f"Ubicación: {DEST_BASE}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Selecciona imágenes aleatorias para dataset de entrenamiento"
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        help=f"Número de imágenes por clase (default: {IMAGES_PER_CLASS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Semilla para aleatorización (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Solo mostrar la selección sin copiar archivos"
    )

    args = parser.parse_args()

    # Configurar semilla para reproducibilidad
    random.seed(args.seed)

    print("=" * 60)
    print("SELECCIÓN DE DATOS PARA DATASET DE ENTRENAMIENTO")
    print("=" * 60)
    print(f"Objetivo: {args.images_per_class} imágenes por clase")
    print(f"Semilla aleatoria: {args.seed}")
    print(f"Directorio origen: {SOURCE_BASE}")
    print(f"Directorio destino: {DEST_BASE}")

    if args.dry_run:
        print("MODO DRY-RUN: Solo se mostrará la selección, no se copiarán archivos")

    # Verificar que existen los directorios origen
    for class_name in CLASSES:
        class_path = os.path.join(SOURCE_BASE, class_name)
        if not os.path.exists(class_path):
            print(f"ERROR: No existe el directorio {class_path}")
            return 1

    # Seleccionar imágenes de cada clase
    selected_images = {}

    for class_name in CLASSES:
        print("\n" + "-" * 40)
        selected_images[class_name] = select_images_from_class(class_name, args.images_per_class)

    if args.dry_run:
        print("\n" + "=" * 60)
        print("RESUMEN DE SELECCIÓN (DRY-RUN)")
        print("=" * 60)
        for class_name, images in selected_images.items():
            print(f"Clase '{class_name}': {len(images)} imágenes seleccionadas")
        return 0

    # Crear estructura de directorios
    print("\n" + "-" * 40)
    print("Creando estructura de directorios...")
    create_destination_structure()

    # Copiar archivos seleccionados
    copy_selected_images(selected_images)

    # Mostrar resumen final
    show_final_summary()

    return 0


if __name__ == "__main__":
    exit(main())
