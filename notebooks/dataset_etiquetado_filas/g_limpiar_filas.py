import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

BASE_DIR = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/b_zonas"
NOMBRE_CARPETA_COLUMNAS = "colyolo_columna"
PREFIX_CARPETA_FILAS = "fil_"
NUM_WORKERS = 6


def buscar_carpetas_columnas_en_dir(dir_nivel1):
    """
    Busca carpetas de columnas dentro de un directorio de nivel 1.
    Función auxiliar para paralelización.
    """
    carpetas_encontradas = []
    if not dir_nivel1.is_dir():
        return carpetas_encontradas

    for dir_nivel2 in dir_nivel1.iterdir():
        if dir_nivel2.is_dir() and NOMBRE_CARPETA_COLUMNAS in dir_nivel2.name:
            carpetas_encontradas.append(dir_nivel2)

    return carpetas_encontradas


def buscar_filas_en_carpeta_columna(carpeta_columna):
    """
    Busca carpetas con archivos de filas dentro de una carpeta de columnas.
    Función auxiliar para paralelización.
    """
    carpetas_filas = []

    for subcarpeta in carpeta_columna.iterdir():
        if not subcarpeta.is_dir():
            continue

        # Verificar si la carpeta contiene archivos con el prefijo de filas
        archivos_filas = list(subcarpeta.glob(f"{PREFIX_CARPETA_FILAS}*.png"))

        if len(archivos_filas) > 0:
            carpetas_filas.append(
                {
                    "path": subcarpeta,
                    "num_filas": len(archivos_filas),
                    "parent": carpeta_columna,
                }
            )

    return carpetas_filas


def eliminar_carpeta_filas(info):
    """
    Elimina una carpeta de filas y su contenido.
    Función auxiliar para paralelización.
    """
    try:
        num_filas = info["num_filas"]
        carpeta = info["path"]

        # Eliminar la carpeta y todo su contenido
        shutil.rmtree(carpeta)

        return {
            "success": True,
            "num_filas": num_filas,
            "carpeta": f"{carpeta.parent.name}/{carpeta.name}",
        }

    except Exception as e:
        return {
            "success": False,
            "num_filas": 0,
            "carpeta": str(info["path"]),
            "error": str(e),
        }


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
    print(f"🔧 Usando {NUM_WORKERS} workers en paralelo...")
    print("=" * 80)

    # Obtener todos los directorios de nivel 1
    dirs_nivel1 = [d for d in base_path.iterdir() if d.is_dir()]

    # Buscar carpetas de columnas en paralelo
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(buscar_carpetas_columnas_en_dir, dir_nivel1)
            for dir_nivel1 in dirs_nivel1
        ]

        for future in as_completed(futures):
            carpetas_columnas.extend(future.result())

    print(f"✅ Se encontraron {len(carpetas_columnas)} carpetas con '{NOMBRE_CARPETA_COLUMNAS}'")

    # Buscar carpetas de filas dentro de cada carpeta de columnas en paralelo
    carpetas_filas_encontradas = []

    print(f"🔍 Buscando archivos de filas en {len(carpetas_columnas)} carpetas...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(buscar_filas_en_carpeta_columna, carpeta_columna)
            for carpeta_columna in carpetas_columnas
        ]

        for future in as_completed(futures):
            carpetas_filas_encontradas.extend(future.result())

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
        print(f"🔧 Usando {NUM_WORKERS} workers en paralelo...")
        print("=" * 80)

        total_eliminadas = 0
        total_filas_eliminadas = 0
        errores = []

        # Eliminar carpetas en paralelo con barra de progreso
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(eliminar_carpeta_filas, info) for info in carpetas_filas_encontradas
            ]

            with tqdm(
                total=len(carpetas_filas_encontradas),
                desc="🗑️  Eliminando carpetas",
                unit="carpeta",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for future in as_completed(futures):
                    resultado = future.result()
                    if resultado["success"]:
                        total_eliminadas += 1
                        total_filas_eliminadas += resultado["num_filas"]
                    else:
                        errores.append(resultado)
                    pbar.update(1)

        print("\n" + "=" * 80)
        print(
            f"✅ RESUMEN: {total_eliminadas} carpetas eliminadas ({total_filas_eliminadas} archivos de filas)"
        )

        if errores:
            print(f"❌ Errores: {len(errores)} carpetas no pudieron ser eliminadas")
            for error in errores:
                print(f"   └─ {error['carpeta']}: {error.get('error', 'Error desconocido')}")

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
