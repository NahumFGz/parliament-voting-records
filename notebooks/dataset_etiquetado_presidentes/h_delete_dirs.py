import os
import shutil
from pathlib import Path

# ⚙️ Configuraciones
base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
carpeta_a_eliminar = "croped_presidente"

# 🔧 Modo de confirmación
CONFIRMAR_ANTES_ELIMINAR = True  # Si True, pide confirmación antes de eliminar


# ==================== FUNCIONES DE UTILIDAD ====================


def buscar_carpetas_dentro_subcarpetas(base_path, nombre_carpeta):
    """
    Busca carpetas que contengan el texto especificado dentro del primer nivel de subcarpetas.
    Solo busca en las carpetas directamente dentro de base_dir, no recursivamente.

    Estructura de búsqueda:
    - base_dir/subcarpeta1/nombre_carpeta/ ✓
    - base_dir/subcarpeta1/algo_nombre_carpeta/ ✓
    - base_dir/subcarpeta1/nombre_carpeta_algo/ ✓
    - base_dir/subcarpeta1/otra_carpeta/nombre_carpeta/ ✗ (profundidad 2)
    - base_dir/nombre_carpeta/ ✗ (no está dentro de una subcarpeta)

    Args:
        base_path: Path del directorio base
        nombre_carpeta: Texto que debe contener el nombre de la carpeta

    Returns:
        Lista de Path con todas las carpetas encontradas
    """
    carpetas_encontradas = []

    # Buscar solo en el primer nivel de subcarpetas (base_dir/*)
    for subcarpeta in base_path.iterdir():
        if subcarpeta.is_dir():
            # Buscar carpetas que contengan el texto en su nombre
            for item in subcarpeta.iterdir():
                if item.is_dir() and nombre_carpeta in item.name:
                    carpetas_encontradas.append(item)

    return sorted(carpetas_encontradas)


def mostrar_resumen_carpetas(carpetas):
    """
    Muestra un resumen de las carpetas encontradas.

    Args:
        carpetas: Lista de Path con carpetas encontradas

    Returns:
        True si hay carpetas, False si está vacía
    """
    if not carpetas:
        print("\n✅ No se encontraron carpetas para eliminar")
        return False

    print(f"\n📊 Se encontraron {len(carpetas)} carpeta(s) que contienen '{carpeta_a_eliminar}':")

    # Mostrar las primeras 10 carpetas
    print("\n📂 Carpetas encontradas (mostrando las primeras 10):")
    for i, carpeta in enumerate(carpetas[:10], start=1):
        # Mostrar ruta relativa desde base_dir
        try:
            ruta_relativa = carpeta.relative_to(Path(base_dir))
            print(f"   {i}. {ruta_relativa}")
        except ValueError:
            print(f"   {i}. {carpeta}")

    if len(carpetas) > 10:
        print(f"   ... y {len(carpetas) - 10} carpeta(s) más")

    return True


def solicitar_confirmacion():
    """
    Solicita confirmación al usuario antes de eliminar.

    Returns:
        True si confirma, False si cancela
    """
    print("\n" + "⚠️ " * 20)
    print("⚠️  Esta acción eliminará permanentemente las carpetas")
    print("⚠️ " * 20)
    respuesta = input(f"\n¿Deseas continuar? (escribe 'SI' para confirmar): ").strip()

    return respuesta.upper() == "SI"


def eliminar_carpetas(carpetas):
    """
    Elimina las carpetas especificadas.

    Args:
        carpetas: Lista de Path con carpetas a eliminar

    Returns:
        Tupla (exitosas, fallidas) con contadores
    """
    exitosas = 0
    fallidas = 0

    print("\n🗑️  Iniciando eliminación...")

    for i, carpeta in enumerate(carpetas, start=1):
        try:
            shutil.rmtree(carpeta)
            exitosas += 1

            # Mostrar progreso cada 50 carpetas o en la primera
            if i % 50 == 0 or i == 1:
                print(f"📊 Progreso: {i}/{len(carpetas)} carpetas eliminadas")

        except Exception as e:
            fallidas += 1
            print(f"❌ Error al eliminar {carpeta}: {e}")

    return exitosas, fallidas


def mostrar_resumen_final(exitosas, fallidas, total):
    """
    Muestra un resumen final de la operación.

    Args:
        exitosas: Número de carpetas eliminadas exitosamente
        fallidas: Número de carpetas que fallaron al eliminar
        total: Total de carpetas procesadas
    """
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE ELIMINACIÓN")
    print("=" * 60)
    print(f"✅ Eliminadas exitosamente: {exitosas}")
    if fallidas > 0:
        print(f"❌ Fallidas: {fallidas}")
    print(f"📦 Total procesadas: {total}")
    print("=" * 60)


# ==================== PROGRAMA PRINCIPAL ====================

if __name__ == "__main__":
    print("=" * 70)
    print("🗑️  SCRIPT DE ELIMINACIÓN DE CARPETAS")
    print("=" * 70)
    print(f"\n📂 Directorio base: {base_dir}")
    print(f"🏷️  Carpeta a buscar: {carpeta_a_eliminar}")

    base_path = Path(base_dir)

    # Validar que el directorio base existe
    if not base_path.exists():
        print(f"❌ Error: El directorio {base_dir} no existe")
        exit(1)

    # 📋 Paso 1: Buscar carpetas dentro de subcarpetas (solo primer nivel)
    print(f"\n🔍 Buscando carpetas que contengan '{carpeta_a_eliminar}' en: {base_dir}/*/")
    carpetas_encontradas = buscar_carpetas_dentro_subcarpetas(base_path, carpeta_a_eliminar)

    # 📊 Paso 2: Mostrar resumen
    hay_carpetas = mostrar_resumen_carpetas(carpetas_encontradas)

    if not hay_carpetas:
        exit(0)

    # ⚠️ Paso 3: Solicitar confirmación (si está activado)
    if CONFIRMAR_ANTES_ELIMINAR:
        if not solicitar_confirmacion():
            print("\n🚫 Operación cancelada por el usuario")
            exit(0)

    # 🗑️ Paso 4: Eliminar carpetas
    exitosas, fallidas = eliminar_carpetas(carpetas_encontradas)

    # ✅ Paso 5: Mostrar resumen final
    mostrar_resumen_final(exitosas, fallidas, len(carpetas_encontradas))

    if fallidas == 0:
        print("\n✅ Todas las carpetas fueron eliminadas exitosamente")
    else:
        print(f"\n⚠️ Se completó con {fallidas} error(es)")
