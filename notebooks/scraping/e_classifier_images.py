import os
import shutil

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# ========== VARIABLES GLOBALES ==========
INPUT_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/data/scraping/b_extract_images"
OUTPUT_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/data/scraping/c_classified"

MODEL_NAME = "resnet50"
CLASS_NAMES = ["asistencia", "otros", "votacion"]
MODEL_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/notebooks/scraping/final_model_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_images_in_path(
    path, recursive=True, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff")
):
    """
    Lista todas las imágenes en una carpeta dada.
    """
    image_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
        if not recursive:
            break
    return image_paths


def get_input_size(model_name):
    if model_name == "inception_v3":
        return (299, 299)
    return (224, 224)


def get_model(model_name, num_classes):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado en este script.")
    return model


def classify_and_save(image_path, model, transform, class_names, output_base_path, device):
    """
    Clasifica una imagen y la guarda en la carpeta correspondiente según su clase.

    Retorna:
    - predicted_class: nombre de la clase predicha
    """
    # Preparar imagen
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]

    # Crear carpeta de destino si no existe
    output_class_path = os.path.join(output_base_path, predicted_class)
    os.makedirs(output_class_path, exist_ok=True)

    # Obtener nombre del archivo y guardar en la nueva ubicación
    filename = os.path.basename(image_path)
    output_file_path = os.path.join(output_class_path, filename)

    # Copiar archivo a la nueva ubicación
    shutil.copy2(image_path, output_file_path)

    return predicted_class


def main():
    """
    Función principal que clasifica todas las imágenes del INPUT_PATH
    y las guarda organizadas por clase en OUTPUT_PATH.
    """
    print(f"🚀 Iniciando clasificación...")
    print(f"📂 Input: {INPUT_PATH}")
    print(f"📁 Output: {OUTPUT_PATH}")
    print(f"🤖 Modelo: {MODEL_NAME}")
    print(f"🏷️  Clases: {CLASS_NAMES}")
    print(f"💻 Device: {DEVICE}\n")

    # Crear carpeta de salida si no existe
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Listar imágenes
    images = list_images_in_path(INPUT_PATH)
    total_images = len(images)
    print(f"📸 Total de imágenes encontradas: {total_images}\n")

    if total_images == 0:
        print("⚠️  No se encontraron imágenes en el INPUT_PATH")
        return

    # Preparar modelo
    num_classes = len(CLASS_NAMES)
    input_size = get_input_size(MODEL_NAME)

    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model = get_model(MODEL_NAME, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("✅ Modelo cargado correctamente\n")
    print("=" * 60)

    # Clasificar y guardar cada imagen
    class_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for idx, image_path in enumerate(images, 1):
        try:
            predicted_class = classify_and_save(
                image_path=image_path,
                model=model,
                transform=transform,
                class_names=CLASS_NAMES,
                output_base_path=OUTPUT_PATH,
                device=DEVICE,
            )

            class_counts[predicted_class] += 1

            print(f"[{idx}/{total_images}] {os.path.basename(image_path)} → {predicted_class}")

        except Exception as e:
            print(f"❌ Error procesando {image_path}: {str(e)}")

    # Resumen final
    print("=" * 60)
    print("\n📊 RESUMEN DE CLASIFICACIÓN:")
    print("-" * 40)
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"  {class_name:15} : {count:4} imágenes ({percentage:5.1f}%)")
    print("-" * 40)
    print(f"  {'TOTAL':15} : {total_images:4} imágenes")
    print("\n✅ Clasificación completada!")


if __name__ == "__main__":
    main()
