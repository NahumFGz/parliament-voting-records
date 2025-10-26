import os

import yaml
from ultralytics import YOLO

# Colores ANSI para resaltar
RESET = "\033[0m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"

# Configuración de rutas
CONFIG_PATH = "experiments_yolo.yml"
BASE_DATA_PATH = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/publicdata-yolo-ocr/data/g_labelstudio_split_folds/votacion"
PROJECT_DIR = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/publicdata-yolo-ocr/validation_ic/yolo/experimentos_votacion"

# Cargar configuración de experimentos
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Filtrar experimentos a ejecutar
experiments = [exp for exp in config["experiments"] if exp.get("execute", True)]
total_experiments = len(experiments)

# Definir los folds disponibles
folds = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
total_folds = len(folds)

# Calcular total de ejecuciones
total_runs = total_experiments * total_folds

print(f"{BOLD}{BLUE}🚀 Iniciando validación cruzada con {total_folds} folds{RESET}")
print(f"{BOLD}{CYAN}📊 Total de experimentos: {total_experiments}{RESET}")
print(f"{BOLD}{MAGENTA}🔄 Total de ejecuciones: {total_runs}{RESET}")
print(f"{BOLD}{'='*60}{RESET}")

run_counter = 0

# Iterar sobre cada fold
for fold_idx, fold in enumerate(folds, start=1):
    fold_data_path = os.path.join(BASE_DATA_PATH, fold, "dataset.yml")

    print(f"\n{BOLD}{BLUE}📁 FOLD {fold_idx}/{total_folds}: {fold}{RESET}")
    print(f"{CYAN}📄 Dataset: {fold_data_path}{RESET}")

    # Verificar que existe el archivo de dataset del fold
    if not os.path.exists(fold_data_path):
        print(f"{BOLD}❌ ERROR: No se encontró el dataset para {fold} en {fold_data_path}{RESET}")
        continue

    # Iterar sobre cada experimento para este fold
    for exp_idx, exp in enumerate(experiments, start=1):
        run_counter += 1
        name = exp["name"]
        params = exp["params"]

        # Crear nombre del experimento incluyendo el fold
        experiment_name = f"{name}_{fold}"

        print(
            f"\n{BOLD}{YELLOW}[🔁 {run_counter}/{total_runs}] Experimento: {experiment_name}{RESET}"
        )
        print(
            f"{CYAN}   └─ Fold: {fold} | Experimento: {exp_idx}/{total_experiments} | Modelo: {params['weights']}{RESET}"
        )
        print(
            f"{CYAN}   └─ Parámetros: imgsz={params['imgsz']}, batch={params['batch']}, epochs={params['epochs']}{RESET}"
        )

        try:
            # Crear y entrenar el modelo
            model = YOLO(params["weights"])
            model.train(
                data=fold_data_path,
                epochs=params["epochs"],
                imgsz=params["imgsz"],
                batch=params["batch"],
                device=params["device"],
                patience=params["patience"],
                project=PROJECT_DIR,
                name=experiment_name,
                verbose=True,
            )

            print(f"{GREEN}   ✅ Completado: {experiment_name}{RESET}")

        except Exception as e:
            print(f"{BOLD}❌ ERROR en {experiment_name}: {str(e)}{RESET}")
            continue

    print(f"{BOLD}{GREEN}✅ Completados todos los experimentos para {fold}{RESET}")

print(f"\n{BOLD}{GREEN}🎉 VALIDACIÓN CRUZADA COMPLETADA{RESET}")
print(f"{BOLD}{GREEN}📊 Total de ejecuciones realizadas: {run_counter}/{total_runs}{RESET}")
print(f"{BOLD}{GREEN}📁 Resultados guardados en: {PROJECT_DIR}{RESET}")
print(f"{BOLD}{'='*60}{RESET}")
