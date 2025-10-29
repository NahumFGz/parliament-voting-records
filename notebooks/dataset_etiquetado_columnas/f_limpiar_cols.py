import os
from pathlib import Path

PREFIX = "colyolo_"
base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"

# Extensiones de imagen comunes
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def eliminar_imagenes_con_prefijo(base_dir, prefix):
    """
    Recorre todas las subcarpetas de base_dir y elimina las imágenes
    que contienen el prefijo especificado en su nombre.

    Args:
        base_dir: Directorio base donde buscar
        prefix: Prefijo a buscar en los nombres de archivo
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: El directorio {base_dir} no existe")
        return

    contador_eliminados = 0
    archivos_eliminados = []

    # Recorrer recursivamente todas las subcarpetas
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            # Verificar si el archivo es una imagen
            file_path = Path(root) / filename
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                # Verificar si el nombre contiene el prefijo
                if prefix in filename:
                    try:
                        file_path.unlink()
                        contador_eliminados += 1
                        archivos_eliminados.append(str(file_path))
                        print(f"Eliminado: {file_path}")
                    except Exception as e:
                        print(f"Error al eliminar {file_path}: {e}")

    print(f"\n{'='*60}")
    print(f"Resumen:")
    print(f"Total de imágenes eliminadas: {contador_eliminados}")
    print(f"{'='*60}")

    return archivos_eliminados


if __name__ == "__main__":
    print(f"Buscando imágenes con prefijo '{PREFIX}' en: {base_dir}")
    print(f"{'='*60}\n")

    # Preguntar confirmación antes de eliminar
    respuesta = input(
        f"¿Estás seguro de que quieres eliminar todas las imágenes que contengan '{PREFIX}'? (s/n): "
    )

    if respuesta.lower() in ["s", "si", "sí", "yes", "y"]:
        archivos = eliminar_imagenes_con_prefijo(base_dir, PREFIX)
    else:
        print("Operación cancelada.")
