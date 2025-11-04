import csv
import os
import re
from pathlib import Path

import pandas as pd

BASE_DIR = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
OUTPUT_DIR = "/home/nahumfg/GithubProjects/parliament-voting-records/data/col_rows"

SUB_SUB_DIR_PREFIX = "colyolo_columna_"
COL_CONGRESISTAS = "_congresista_"
COL_GRUPO_PARLAMENTARIO = "_grupo_parlamentario_"
COL_VOTOS = "_voto_"


def natural_sort_key(text):
    """
    Clave para ordenamiento natural (fil_1.png, fil_2.png, ..., fil_10.png)
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r"(\d+)", text)]


def extract_dir_name(full_path):
    """Extrae el nombre de directorio padre (ej: 0a3e4a10-ae00-5869-bb60-769e30e79e4c_page003_)"""
    parts = Path(full_path).parts
    # Buscar el directorio que termina con _page###_
    for part in parts:
        if "_page" in part and part.endswith("_"):
            return part
    return None


def extract_column_number(full_path):
    """Extrae el número de columna (ej: columna_1, columna_2, columna_3)"""
    parts = Path(full_path).parts
    for part in parts:
        if part.startswith("colyolo_columna_"):
            # Formato: colyolo_columna_X_Y, queremos el primer número X
            match = re.search(r"colyolo_columna_(\d+)_", part)
            if match:
                return f"columna_{match.group(1)}"
    return None


def extract_column_type(full_path):
    """Extrae el tipo de columna (congresista, grupo_parlamentario, voto)"""
    parts = Path(full_path).parts
    for part in parts:
        if part.startswith("colcolyolo_"):
            if "_congresista_" in part:
                return "congresista"
            elif "_grupo_parlamentario_" in part:
                return "grupo_parlamentario"
            elif "_voto_" in part:
                return "voto"
    return None


def find_images_in_category(base_dir, sub_sub_dir_prefix, category_name, debug=False):
    """
    Busca imágenes fil_*.png en carpetas que contengan category_name

    Retorna una lista de tuplas (dir_name, column, column_type, image_path, image_name)
    """
    results = []

    # Recorrer todas las subcarpetas en BASE_DIR
    for root, dirs, files in os.walk(base_dir):
        basename = os.path.basename(root)

        # Las carpetas con imágenes se llaman colcolyolo_congresista_*, colcolyolo_voto_*, etc.
        # Verificar si la carpeta contiene el category_name
        if category_name in basename and basename.startswith("colcolyolo"):
            if debug:
                print(f"  DEBUG: Carpeta con category encontrada: {basename}")
            # Buscar archivos fil_*.png
            for file in files:
                if file.startswith("fil_") and file.endswith(".png"):
                    full_path = os.path.join(root, file)
                    dir_name = extract_dir_name(full_path)
                    column = extract_column_number(full_path)
                    column_type = extract_column_type(full_path)
                    if dir_name and column and column_type:
                        results.append((dir_name, column, column_type, full_path, file))

    return results


def save_to_parquet(data, output_name, output_dir, create_sample=True, sample_size=1000):
    """Guarda los datos en formato parquet y crea una muestra CSV"""
    # Ordenar por dir_name, column, column_type y luego por image_name (con ordenamiento natural)
    # data: (dir_name, column, column_type, image_path, image_name)
    sorted_data = sorted(data, key=lambda x: (x[0], x[1], x[2], natural_sort_key(x[4])))

    # Crear DataFrame
    df = pd.DataFrame(
        sorted_data, columns=["dir_name", "column", "column_type", "image_path", "image_name"]
    )

    # Guardar archivo completo en parquet
    parquet_file = os.path.join(output_dir, output_name.replace(".csv", ".parquet"))
    df.to_parquet(parquet_file, index=False)
    print(f"✓ Archivo Parquet creado: {parquet_file} ({len(sorted_data)} imágenes)")

    # Crear archivo de muestra CSV con las primeras N filas
    if create_sample and len(sorted_data) > sample_size:
        sample_file = os.path.join(
            output_dir, output_name.replace(".csv", f"_sample_{sample_size}.csv")
        )
        df.head(sample_size).to_csv(sample_file, index=False)
        print(f"✓ Archivo de muestra CSV creado: {sample_file} ({sample_size} imágenes)")


def main():
    print("Iniciando búsqueda de imágenes...")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}\n")

    # Verificar que BASE_DIR existe
    if not os.path.exists(BASE_DIR):
        print(f"ERROR: El directorio {BASE_DIR} no existe")
        return

    # Crear OUTPUT_DIR si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Directorio de salida verificado/creado: {OUTPUT_DIR}\n")

    # Buscar imágenes para cada categoría
    print("1. Buscando imágenes de CONGRESISTAS...")
    congresistas_data = find_images_in_category(
        BASE_DIR, SUB_SUB_DIR_PREFIX, COL_CONGRESISTAS, debug=False
    )
    save_to_parquet(congresistas_data, "congresistas_images.csv", OUTPUT_DIR)

    print("\n2. Buscando imágenes de GRUPO PARLAMENTARIO...")
    grupo_parlamentario_data = find_images_in_category(
        BASE_DIR, SUB_SUB_DIR_PREFIX, COL_GRUPO_PARLAMENTARIO
    )
    save_to_parquet(grupo_parlamentario_data, "grupo_parlamentario_images.csv", OUTPUT_DIR)

    print("\n3. Buscando imágenes de VOTOS...")
    votos_data = find_images_in_category(BASE_DIR, SUB_SUB_DIR_PREFIX, COL_VOTOS)
    save_to_parquet(votos_data, "votos_images.csv", OUTPUT_DIR)

    # Generar CSV con los dir_name únicos
    print("\n4. Generando archivo de dir_names únicos...")
    all_dir_names = set()
    for data in [congresistas_data, grupo_parlamentario_data, votos_data]:
        for row in data:
            all_dir_names.add(row[0])  # dir_name es el primer elemento

    # Ordenar los dir_names
    sorted_dir_names = sorted(all_dir_names)

    # Guardar en CSV en OUTPUT_DIR
    dir_names_file = os.path.join(OUTPUT_DIR, "dir_names.csv")
    with open(dir_names_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dir_name"])
        for dir_name in sorted_dir_names:
            writer.writerow([dir_name])

    print(f"✓ Archivo creado: {dir_names_file} ({len(sorted_dir_names)} directorios únicos)")

    print("\n" + "=" * 50)
    print("¡Proceso completado!")
    print(f"Total de imágenes encontradas:")
    print(f"  - Congresistas: {len(congresistas_data)}")
    print(f"  - Grupo Parlamentario: {len(grupo_parlamentario_data)}")
    print(f"  - Votos: {len(votos_data)}")
    print(f"\nTotal de directorios únicos: {len(sorted_dir_names)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
