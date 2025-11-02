import shutil
from pathlib import Path

# ⚙️ Configuraciones
base_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"

# 🏷️ Prefijo de las carpetas a eliminar (carpetas creadas por el script anterior)
PREFIX = "colyolo_"  # Elimina carpetas que empiezan con este prefijo

# 🔧 Modo de confirmación
CONFIRMAR_ANTES_ELIMINAR = True  # Si True, pide confirmación antes de eliminar


# ==================== FUNCIONES DE UTILIDAD ====================


def buscar_carpetas_con_prefix(base_path, prefix):
    """
    Busca recursivamente todas las carpetas que empiezan con el prefijo.

    Args:
        base_path: Path del directorio base
        prefix: Prefijo a buscar al inicio del nombre de carpeta

    Returns:
        Lista de Path con todas las carpetas encontradas
    """
    carpetas = []

    # Buscar todas las carpetas recursivamente
    for item in base_path.rglob("*"):
        if item.is_dir() and item.name.startswith(prefix):
            carpetas.append(item)

    return sorted(carpetas)


def mostrar_resumen_carpetas(carpetas):
    """
    Muestra un resumen de las carpetas encontradas.

    Args:
        carpetas: Lista de Path con carpetas
    """
    if not carpetas:
        print("✅ No se encontraron carpetas con el prefijo especificado")
        return False

    print(f"\n📋 Se encontraron {len(carpetas)} carpetas que empiezan con '{PREFIX}'")
    print("\n🔍 Primeras 10 carpetas a eliminar:")
    for carpeta in carpetas[:10]:
        print(f"   📁 {carpeta}")

    if len(carpetas) > 10:
        print(f"   ... y {len(carpetas) - 10} carpetas más")

    return True


def solicitar_confirmacion():
    """
    Solicita confirmación del usuario antes de eliminar.

    Returns:
        bool: True si el usuario confirma, False en caso contrario
    """
    print("\n⚠️  ADVERTENCIA: Esta acción NO se puede deshacer.")
    respuesta = input("¿Deseas continuar con la eliminación? (SI/no): ").strip()
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

            # Mostrar progreso cada 100 carpetas
            if i % 100 == 0 or i == 1:
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
        fallidas: Número de carpetas que fallaron al eliminarse
        total: Total de carpetas procesadas
    """
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL")
    print("=" * 60)
    print(f"✅ Carpetas eliminadas exitosamente: {exitosas}")
    if fallidas > 0:
        print(f"❌ Carpetas con errores: {fallidas}")
    print(f"📦 Total procesadas: {total}")
    print("=" * 60)


# ==================== PROGRAMA PRINCIPAL ====================

if __name__ == "__main__":
    print("🔍 Buscando carpetas a eliminar...")
    print(f"📂 Directorio base: {base_dir}")
    print(f"🏷️  Prefijo a buscar: {PREFIX}")

    base_path = Path(base_dir)

    # Validar que el directorio base existe
    if not base_path.exists():
        print(f"❌ Error: El directorio {base_dir} no existe")
        exit(1)

    # 📋 Paso 1: Buscar carpetas con el prefijo
    carpetas_encontradas = buscar_carpetas_con_prefix(base_path, PREFIX)

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
