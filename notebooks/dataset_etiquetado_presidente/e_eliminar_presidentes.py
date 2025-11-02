import os
from pathlib import Path

# ⚙️ Configuraciones
base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"

# 🏷️ Prefijo de las imágenes a eliminar
PREFIX_CROPED = "presidente_"


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


def eliminar_imagenes(imagenes):
    """
    Elimina las imágenes especificadas.

    Args:
        imagenes: Lista de Path con las imágenes a eliminar

    Returns:
        Tupla (cantidad_eliminadas, cantidad_errores)
    """
    eliminadas = 0
    errores = 0

    for img_path in imagenes:
        try:
            if img_path.exists():
                os.remove(str(img_path))
                eliminadas += 1
        except Exception as e:
            errores += 1
            print(f"   ⚠️ Error al eliminar {img_path.name}: {e}")

    return eliminadas, errores


# ==================== PROGRAMA PRINCIPAL ====================

if __name__ == "__main__":
    print("=" * 70)
    print("🗑️  SCRIPT DE LIMPIEZA DE IMÁGENES GENERADAS")
    print("=" * 70)

    # 📋 Paso 1: Buscar todas las imágenes
    print(f"\n🔍 Buscando imágenes con prefijo '{PREFIX_CROPED}'...")
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"❌ Error: El directorio {base_dir} no existe")
        exit(1)

    todas_imagenes = buscar_todas_imagenes(base_path)
    imagenes_a_eliminar = filtrar_imagenes_con_prefijo(todas_imagenes, PREFIX_CROPED)
    total_eliminar = len(imagenes_a_eliminar)

    # ⚠️ Validar que hay imágenes para eliminar
    if total_eliminar == 0:
        print(f"✅ No se encontraron imágenes con '{PREFIX_CROPED}' en el nombre")
        print("   No hay nada que eliminar")
        exit(0)

    # 📊 Mostrar resumen
    print(f"\n📊 Se encontraron {total_eliminar} imágenes para eliminar")

    # 🔒 Confirmación
    print("\n" + "⚠️ " * 23)
    print("⚠️  Esta acción eliminará permanentemente los archivos")
    print("⚠️ " * 23)
    respuesta = input(f"\n¿Deseas eliminar {total_eliminar} imágenes? (S/n): ").strip().lower()

    if respuesta not in ["s", "si", "sí", "yes", "y", ""]:
        print("\n❌ Operación cancelada por el usuario")
        exit(0)

    # 🗑️ Eliminar imágenes
    print(f"\n🗑️  Eliminando {total_eliminar} imágenes...")
    eliminadas, errores = eliminar_imagenes(imagenes_a_eliminar)

    # 📈 Mostrar resultados
    print(f"\n{'=' * 70}")
    print(f"✅ Eliminación completada:")
    print(f"   ✓ Eliminadas: {eliminadas} imágenes")
    if errores > 0:
        print(f"   ✗ Errores: {errores} imágenes")
    print(f"{'=' * 70}")
