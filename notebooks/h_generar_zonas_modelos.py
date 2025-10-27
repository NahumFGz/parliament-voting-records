import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# ⚙️ Configuraciones para votación
input_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/c_enderezados"
output_dir = "/home/nahumfg/GithubProjects/parliament-voting-records/data/procesamiento_todas_votaciones/d_divisiones"
model_path = "/home/nahumfg/GithubProjects/parliament-voting-records/validation/yolo/experiments/yolo11n_img480_bs32_fold_9/weights/best.pt"

# 🧠 Cargar modelo YOLO
model = YOLO(model_path)

# 📂 Crear carpeta destino si no existe
os.makedirs(output_dir, exist_ok=True)

# 📋 Listar imágenes válidas
img_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
total_imgs = len(img_files)
print(f"📦 Total de imágenes detectadas: {total_imgs}")

# 🔁 Procesar imágenes
for idx, img_file in enumerate(img_files, start=1):
    if idx % 100 == 0 or idx == 1:
        print(f"\n📊 Progreso: {idx}/{total_imgs} imágenes procesadas")

    image_path = os.path.join(input_dir, img_file)
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"[⚠️] No se pudo leer la imagen: {image_path}")
        continue

    # 📍 Predecir zonas
    results = model.predict(source=image_bgr, conf=0.25)
    for result in results:
        detecciones = result.boxes
        labels = result.names

        # 📁 Crear carpeta con nombre base de la imagen
        base_name = os.path.splitext(img_file)[0]
        img_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(img_output_dir, exist_ok=True)

        for i, box in enumerate(detecciones):
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            label = labels[int(box.cls[0])]

            # Recortar zona
            zona = image_bgr[y_min:y_max, x_min:x_max]

            # Guardar recorte
            zona_path = os.path.join(img_output_dir, f"{label}_{i+1}.png")
            cv2.imwrite(zona_path, zona)
