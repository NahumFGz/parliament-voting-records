import shutil
from pathlib import Path

BASE_DIR = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
NOMBRE_CARPETA_COLUMNAS = "colyolo_columna"
PREFIX_CARPETA_FILAS = "fil_"


def encontrar_carpetas_filas(modo="listar"):
    """
    Encuentra todas las carpetas que contienen filas generadas.

    Parámetros:
    -----------
    modo : str
        - "listar": Solo muestra las carpetas que se encontraron
        - "eliminar": Elimina las carpetas encontradas

    Retorna:
    --------
    int : Número de carpetas encontradas/eliminadas
    """

    base_path = Path(BASE_DIR)

    if not base_path.exists():
        print(f"❌ Error: No se encuentra el directorio BASE_DIR: {BASE_DIR}")
        return 0

    # Buscar todas las carpetas que contengan NOMBRE_CARPETA_COLUMNAS
    carpetas_columnas = []

    print(f"🔍 Buscando carpetas que contengan '{NOMBRE_CARPETA_COLUMNAS}'...")
    print(f"🔍 En el directorio: {BASE_DIR}...")
    print("=" * 80)

    # Recorrer BASE_DIR/xxx (primer nivel)
    for dir_nivel1 in base_path.iterdir():
        if not dir_nivel1.is_dir():
            continue

        # Dentro de cada carpeta, buscar carpetas que contengan NOMBRE_CARPETA_COLUMNAS
        for dir_nivel2 in dir_nivel1.iterdir():
            if dir_nivel2.is_dir() and NOMBRE_CARPETA_COLUMNAS in dir_nivel2.name:
                carpetas_columnas.append(dir_nivel2)

    print(f"✅ Se encontraron {len(carpetas_columnas)} carpetas con '{NOMBRE_CARPETA_COLUMNAS}'")

    # Buscar carpetas de filas dentro de cada carpeta de columnas
    carpetas_filas_encontradas = []

    for carpeta_columna in carpetas_columnas:
        # Buscar subcarpetas que puedan contener filas generadas
        for subcarpeta in carpeta_columna.iterdir():
            if not subcarpeta.is_dir():
                continue

            # Verificar si la carpeta contiene archivos con el prefijo de filas
            archivos_filas = list(subcarpeta.glob(f"{PREFIX_CARPETA_FILAS}*.png"))

            if len(archivos_filas) > 0:
                carpetas_filas_encontradas.append(
                    {
                        "path": subcarpeta,
                        "num_filas": len(archivos_filas),
                        "parent": carpeta_columna,
                    }
                )

    # Mostrar resultados
    print(f"\n📊 Se encontraron {len(carpetas_filas_encontradas)} carpetas con filas generadas")
    print("=" * 80)

    if len(carpetas_filas_encontradas) == 0:
        print("ℹ️  No se encontraron carpetas con filas para eliminar")
        return 0

    # Agrupar por carpeta padre
    carpetas_por_padre = {}
    for info in carpetas_filas_encontradas:
        padre = str(info["parent"])
        if padre not in carpetas_por_padre:
            carpetas_por_padre[padre] = []
        carpetas_por_padre[padre].append(info)

    # Listar o eliminar
    if modo == "listar":
        print("\n📋 CARPETAS QUE SE ELIMINARÍAN:")
        print("=" * 80)

        for padre, carpetas in carpetas_por_padre.items():
            print(f"\n📁 {Path(padre).name}")
            total_filas = 0
            for info in carpetas:
                print(f"   └─ {info['path'].name}/ ({info['num_filas']} filas)")
                total_filas += info["num_filas"]
            print(f"   Total: {len(carpetas)} carpetas, {total_filas} archivos de filas")

        print("\n" + "=" * 80)
        print(f"⚠️  TOTAL: {len(carpetas_filas_encontradas)} carpetas serían eliminadas")
        print("=" * 80)
        print("\n💡 Para eliminar estas carpetas, ejecuta:")
        print("   limpiar_carpetas_filas(modo='eliminar')")

    elif modo == "eliminar":
        print("\n🗑️  ELIMINANDO CARPETAS:")
        print("=" * 80)

        total_eliminadas = 0
        total_filas_eliminadas = 0

        for info in carpetas_filas_encontradas:
            try:
                num_filas = info["num_filas"]
                carpeta = info["path"]

                # Eliminar la carpeta y todo su contenido
                shutil.rmtree(carpeta)

                print(f"✅ Eliminada: {carpeta.parent.name}/{carpeta.name}/ ({num_filas} filas)")
                total_eliminadas += 1
                total_filas_eliminadas += num_filas

            except Exception as e:
                print(f"❌ Error eliminando {carpeta}: {str(e)}")

        print("\n" + "=" * 80)
        print(
            f"✅ RESUMEN: {total_eliminadas} carpetas eliminadas ({total_filas_eliminadas} archivos de filas)"
        )
        print("=" * 80)

    return len(carpetas_filas_encontradas)


def limpiar_carpetas_filas(modo="listar", confirmar=True):
    """
    Función principal para limpiar carpetas de filas.

    Parámetros:
    -----------
    modo : str
        - "listar": Solo muestra las carpetas que se encontraron
        - "eliminar": Elimina las carpetas encontradas
    confirmar : bool
        Si True, pide confirmación antes de eliminar (solo en modo "eliminar")
    """

    if modo == "eliminar" and confirmar:
        print("\n⚠️  ADVERTENCIA: Esta acción eliminará permanentemente las carpetas y archivos.")
        print("⚠️  Esta operación NO se puede deshacer.")
        respuesta = input(
            "\n¿Estás seguro de que quieres continuar? (escribe 'SI' para confirmar): "
        )

        if respuesta.strip().upper() != "SI":
            print("\n❌ Operación cancelada")
            return

        print("\n🗑️  Procediendo con la eliminación...")

    encontrar_carpetas_filas(modo=modo)


# ============= EJEMPLOS DE USO =============

if __name__ == "__main__":
    # Opción 1: Solo listar qué se eliminaría (recomendado primero)
    # print("=" * 80)
    # print("MODO: LISTAR (sin eliminar nada)")
    # print("=" * 80)
    # limpiar_carpetas_filas(modo="listar")

    # Opción 2: Eliminar con confirmación (descomenta para usar)
    print("\n\n")
    print("=" * 80)
    print("MODO: ELIMINAR")
    print("=" * 80)
    limpiar_carpetas_filas(modo="eliminar", confirmar=True)

    # Opción 3: Eliminar sin confirmación (¡PELIGROSO! solo para scripts automatizados)
    # limpiar_carpetas_filas(modo="eliminar", confirmar=False)
