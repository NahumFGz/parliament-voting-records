# # 1. Generar los datasets con K-Fold Cross Validation para entrenar YOLO
import os
import random
import shutil
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_class_distribution(labels_dir: str, image_files: list) -> dict:
    """
    Obtiene la distribución de clases para cada imagen basándose en las etiquetas YOLO.

    Args:
        labels_dir (str): Directorio con archivos de etiquetas .txt
        image_files (list): Lista de nombres de archivos de imagen

    Returns:
        dict: Mapeo de archivo de imagen a clases presentes
    """
    image_classes = {}

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        classes_in_image = set()
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        classes_in_image.add(class_id)

        # Para estratificación, usamos la clase predominante o la primera clase encontrada
        # Si no hay clases, asignamos clase -1
        if classes_in_image:
            image_classes[img_file] = min(classes_in_image)  # Usa la clase con menor ID
        else:
            image_classes[img_file] = -1  # Imagen sin etiquetas

    return image_classes


def create_kfold_datasets(
    images_dir: str,
    labels_dir: str,
    output_base_dir: str,
    dataset_name: str,
    k_folds: int = 5,
    seed: int = 42,
):
    """
    Crea datasets con K-Fold Cross Validation estratificado para YOLO.

    Args:
        images_dir (str): Directorio de imágenes originales
        labels_dir (str): Directorio de etiquetas originales
        output_base_dir (str): Directorio base donde se crearán los folds
        dataset_name (str): Nombre del dataset (asistencia/votacion)
        k_folds (int): Número de folds para cross validation
        seed (int): Semilla para reproducibilidad
    """

    print(f"\n🔄 Creando {k_folds}-fold cross validation para dataset '{dataset_name}'")

    # Crear directorio base si no existe
    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    if os.path.exists(dataset_output_dir):
        shutil.rmtree(dataset_output_dir)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Obtener lista de imágenes
    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    print(f"📊 Total de imágenes encontradas: {len(image_files)}")

    # Obtener distribución de clases para estratificación
    image_classes = get_class_distribution(labels_dir, image_files)

    # Preparar datos para StratifiedKFold
    X = np.array(image_files)
    y = np.array([image_classes[img] for img in image_files])

    # Mostrar distribución de clases
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"📈 Distribución de clases: {dict(zip(unique_classes, counts))}")

    # Crear StratifiedKFold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # Función para copiar archivos
    def copy_files(file_list, dst_img_dir, dst_lbl_dir, set_name, fold_num):
        for fname in file_list:
            # Copiar imagen
            shutil.copy2(os.path.join(images_dir, fname), os.path.join(dst_img_dir, fname))

            # Copiar etiqueta si existe
            label_file = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(dst_lbl_dir, label_file))
            else:
                print(f"[⚠️] Fold {fold_num} - Etiqueta no encontrada para {fname}")

        print(f"✅ Fold {fold_num} - {set_name}: {len(file_list)} imágenes copiadas.")

    # Generar cada fold
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        fold_num = fold_idx + 1
        print(f"\n📁 Procesando Fold {fold_num}/{k_folds}")

        # Crear directorios para este fold
        fold_dir = os.path.join(dataset_output_dir, f"fold_{fold_num}")
        train_images_dir = os.path.join(fold_dir, "train", "images")
        train_labels_dir = os.path.join(fold_dir, "train", "labels")
        val_images_dir = os.path.join(fold_dir, "val", "images")
        val_labels_dir = os.path.join(fold_dir, "val", "labels")

        for path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            os.makedirs(path, exist_ok=True)

        # Obtener archivos para train y val
        train_files = X[train_indices].tolist()
        val_files = X[val_indices].tolist()

        # Copiar archivos
        copy_files(train_files, train_images_dir, train_labels_dir, "Entrenamiento", fold_num)
        copy_files(val_files, val_images_dir, val_labels_dir, "Validación", fold_num)

        # Mostrar distribución de clases en este fold
        train_classes = [image_classes[img] for img in train_files]
        val_classes = [image_classes[img] for img in val_files]

        train_unique, train_counts = np.unique(train_classes, return_counts=True)
        val_unique, val_counts = np.unique(val_classes, return_counts=True)

        print(f"   📊 Train - Clases: {dict(zip(train_unique, train_counts))}")
        print(f"   📊 Val   - Clases: {dict(zip(val_unique, val_counts))}")

    print(f"\n🚀 {k_folds}-fold cross validation completado para '{dataset_name}'!")


def copiar_archivos_generales_folds(
    origen_dir: str, destino_base_dir: str, dataset_name: str, archivos: list[str], k_folds: int = 5
):
    """
    Copia archivos generales como 'classes.txt' o 'notes.json' a cada fold.

    Args:
        origen_dir (str): Ruta de origen donde están los archivos.
        destino_base_dir (str): Directorio base donde están los folds.
        dataset_name (str): Nombre del dataset (asistencia/votacion).
        archivos (list): Lista de nombres de archivos a copiar.
        k_folds (int): Número de folds.
    """
    # Si dataset_name es "" o False, no se agrega
    if dataset_name:
        dataset_dir = os.path.join(destino_base_dir, dataset_name)
    else:
        dataset_dir = destino_base_dir

    for fold_num in range(1, k_folds + 1):
        fold_dir = os.path.join(dataset_dir, f"fold_{fold_num}")

        for file_name in archivos:
            src = os.path.join(origen_dir, file_name)
            dst = os.path.join(fold_dir, file_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"✅ Fold {fold_num} - Copiado '{file_name}' a '{dst}'")
            else:
                print(
                    f"⚠️ [ADVERTENCIA] Fold {fold_num} - '{file_name}' no encontrado en '{origen_dir}'"
                )


def generar_dataset_yaml_folds(
    origen_dir: str,
    destino_base_dir: str,
    dataset_name: str,
    k_folds: int = 5,
    output_file: str = "dataset.yml",
):
    """
    Genera archivos dataset.yml para cada fold con rutas absolutas.

    Args:
        origen_dir (str): Directorio de origen con classes.txt.
        destino_base_dir (str): Directorio base donde están los folds.
        dataset_name (str): Nombre del dataset (asistencia/votacion).
        k_folds (int): Número de folds.
        output_file (str): Nombre del archivo de salida YAML.
    """
    classes_path = os.path.join(origen_dir, "classes.txt")

    if not os.path.exists(classes_path):
        print(
            f"⚠️ [ADVERTENCIA] No se encontró '{classes_path}', no se generaron archivos dataset.yml."
        )
        return

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    names_yaml = "\n".join([f"  - {name}" for name in class_names])

    # Si dataset_name es "" o False, no se agrega
    if dataset_name:
        dataset_dir = os.path.join(destino_base_dir, dataset_name)
    else:
        dataset_dir = destino_base_dir

    for fold_num in range(1, k_folds + 1):
        fold_dir = os.path.join(dataset_dir, f"fold_{fold_num}")

        content = f"""train: {os.path.abspath(os.path.join(fold_dir, "train"))}
val: {os.path.abspath(os.path.join(fold_dir, "val"))}

nc: {len(class_names)}
names:
{names_yaml}
"""

        yaml_path = os.path.join(fold_dir, output_file)
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ Fold {fold_num} - Archivo '{output_file}' creado en '{yaml_path}'")


# A. Crear 10-fold cross validation
create_kfold_datasets(
    images_dir="../../data/dataset_etiquetado_presidente/b_etiquetados/images",
    labels_dir="../../data/dataset_etiquetado_presidente/b_etiquetados/labels",
    output_base_dir="../../data/dataset_etiquetado_presidente/c_split_cross_valid",
    dataset_name="",
    k_folds=10,
    seed=42,
)

# B. Copiar archivos generales para todos los folds
copiar_archivos_generales_folds(
    origen_dir="../../data/dataset_etiquetado_presidente/b_etiquetados",
    destino_base_dir="../../data/dataset_etiquetado_presidente/c_split_cross_valid",
    dataset_name="",
    archivos=["classes.txt", "notes.json"],
    k_folds=10,
)

# C. Generar dataset.yml para todos los folds
generar_dataset_yaml_folds(
    origen_dir="../../data/dataset_etiquetado_presidente/b_etiquetados",
    destino_base_dir="../../data/dataset_etiquetado_presidente/c_split_cross_valid",
    dataset_name="",
    k_folds=10,
)
