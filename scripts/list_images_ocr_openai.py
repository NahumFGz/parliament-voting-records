import csv
import os
from pathlib import Path

base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
output_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/scripts/list_images_ocr_openai"

os.makedirs(output_dir, exist_ok=True)

# Tipos de imágenes a buscar
tipos_imagenes = ["encabezado", "grupo_parlamentario", "resultado", "voto_oral"]

# Extensiones de imagen válidas
extensiones_imagen = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Listas para almacenar los resultados
carpetas_validas = []
carpetas_anomalas = []

# Recorrer todas las subcarpetas
print(f"Explorando carpetas en: {base_dir}")
for subdir in sorted(Path(base_dir).iterdir()):
    if subdir.is_dir():
        # Buscar imágenes por tipo
        imagenes_encontradas = {
            "encabezado": [],
            "grupo_parlamentario": [],
            "resultado": [],
            "voto_oral": [],
        }

        # Buscar todas las imágenes en la carpeta
        for archivo in subdir.iterdir():
            if archivo.is_file() and archivo.suffix.lower() in extensiones_imagen:
                nombre_lower = archivo.name.lower()
                for tipo in tipos_imagenes:
                    if tipo in nombre_lower:
                        imagenes_encontradas[tipo].append(archivo.name)

        # Verificar si la carpeta tiene exactamente 1 de cada tipo
        cantidades = {tipo: len(imagenes) for tipo, imagenes in imagenes_encontradas.items()}
        es_valida = all(cantidad == 1 for cantidad in cantidades.values())

        if es_valida:
            carpetas_validas.append(
                {
                    "DIR_PATH": str(subdir),
                    "DIR_NAME": subdir.name,
                    "ENCABEZADO": imagenes_encontradas["encabezado"][0],
                    "GRUPO_PARLAMENTARIO": imagenes_encontradas["grupo_parlamentario"][0],
                    "RESULTADO": imagenes_encontradas["resultado"][0],
                    "VOTO_ORAL": imagenes_encontradas["voto_oral"][0],
                }
            )
            print(f"✓ {subdir.name}")
        else:
            carpetas_anomalas.append(
                {
                    "DIR_PATH": str(subdir),
                    "DIR_NAME": subdir.name,
                    "ENCABEZADO_COUNT": cantidades["encabezado"],
                    "ENCABEZADO_FILES": (
                        ", ".join(imagenes_encontradas["encabezado"])
                        if imagenes_encontradas["encabezado"]
                        else "NINGUNO"
                    ),
                    "GRUPO_PARLAMENTARIO_COUNT": cantidades["grupo_parlamentario"],
                    "GRUPO_PARLAMENTARIO_FILES": (
                        ", ".join(imagenes_encontradas["grupo_parlamentario"])
                        if imagenes_encontradas["grupo_parlamentario"]
                        else "NINGUNO"
                    ),
                    "RESULTADO_COUNT": cantidades["resultado"],
                    "RESULTADO_FILES": (
                        ", ".join(imagenes_encontradas["resultado"])
                        if imagenes_encontradas["resultado"]
                        else "NINGUNO"
                    ),
                    "VOTO_ORAL_COUNT": cantidades["voto_oral"],
                    "VOTO_ORAL_FILES": (
                        ", ".join(imagenes_encontradas["voto_oral"])
                        if imagenes_encontradas["voto_oral"]
                        else "NINGUNO"
                    ),
                }
            )
            print(f"✗ {subdir.name} - Anómala")

# Guardar CSV de carpetas válidas
csv_validas = os.path.join(output_dir, "carpetas_validas.csv")
with open(csv_validas, "w", newline="", encoding="utf-8") as f:
    if carpetas_validas:
        fieldnames = [
            "DIR_PATH",
            "DIR_NAME",
            "ENCABEZADO",
            "GRUPO_PARLAMENTARIO",
            "RESULTADO",
            "VOTO_ORAL",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(carpetas_validas)

# Guardar CSV de carpetas anómalas
csv_anomalas = os.path.join(output_dir, "carpetas_anomalas.csv")
with open(csv_anomalas, "w", newline="", encoding="utf-8") as f:
    if carpetas_anomalas:
        fieldnames = [
            "DIR_PATH",
            "DIR_NAME",
            "ENCABEZADO_COUNT",
            "ENCABEZADO_FILES",
            "GRUPO_PARLAMENTARIO_COUNT",
            "GRUPO_PARLAMENTARIO_FILES",
            "RESULTADO_COUNT",
            "RESULTADO_FILES",
            "VOTO_ORAL_COUNT",
            "VOTO_ORAL_FILES",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(carpetas_anomalas)

# Imprimir resumen
print(f"\n{'='*60}")
print(f"RESUMEN:")
print(f"{'='*60}")
print(f"Total de carpetas analizadas: {len(carpetas_validas) + len(carpetas_anomalas)}")
print(f"Carpetas válidas (1 de cada tipo): {len(carpetas_validas)}")
print(f"Carpetas anómalas: {len(carpetas_anomalas)}")
print(f"\nArchivos generados:")
print(f"  - {csv_validas} ({len(carpetas_validas)} registros)")
print(f"  - {csv_anomalas} ({len(carpetas_anomalas)} registros)")
