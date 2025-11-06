#!/usr/bin/env python3
"""
Experiment runner con VALIDACIÓN CRUZADA para image-classification models (PyTorch).

MEJORAS PARA RIGOR CIENTÍFICO:
• Implementa validación cruzada k-fold (por defecto k=5)
• Calcula estadísticas robustas (media ± desviación estándar)
• Mantiene un conjunto de test independiente y NO tocado durante CV
• Guarda métricas detalladas por fold y agregadas
• Incluye intervalos de confianza para las métricas

División de datos:
1. Test set (10%) - NUNCA se toca durante entrenamiento/validación
2. Train+Val set (90%) - Se usa para k-fold CV
3. En cada fold: 90% train, 10% val (del conjunto train+val)

YAML schema example:
```yaml
experiments:
  - name: resnet_cv
    params:
      model: resnet50
      lr: 1e-4
      epochs: 15
      batch_size: 32
      patience: 3
      k_folds: 10  # nuevo parámetro
```
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Suprimir warnings de sklearn
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── Paths ─────────────────────────────────────────
EXPERIMENTS_PATH = Path(
    "/home/nahumfg/GithubProjects/parliament-voting-records/validation/classification/experiments_cv.yml"
)
DATA_DIR = Path(
    "/home/nahumfg/GithubProjects/parliament-voting-records/data/dataset_classification/selected"
)
VALIDATION_DIR = Path(
    "/home/nahumfg/GithubProjects/parliament-voting-records/validation/classification/experiments"
)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────── Constants ───────────────────────────────────────
NUM_WORKERS = 10  # Number of workers for data loading
TEST_SIZE = 0.10  # Test set independiente
TRAIN_VAL_SIZE = 0.90  # Para k-fold CV

if not abs(TEST_SIZE + TRAIN_VAL_SIZE - 1.0) < 1e-10:
    raise ValueError(f"Las proporciones deben sumar 1.0, pero suman {TEST_SIZE + TRAIN_VAL_SIZE}")


# ─────────────────────────── Helpers ─────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "GPU"
    return torch.device("cpu"), "CPU"


def get_input_size(model_name: str) -> Tuple[int, int]:
    return (299, 299) if model_name.lower() == "inception_v3" else (224, 224)


def get_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise NotImplementedError(f"Modelo {model_name} no implementado.")
    return model


def load_trained_model(
    experiment_dir: Path, model_name: str, num_classes: int, device: torch.device
):
    """
    Carga un modelo previamente entrenado desde un directorio de experimento.

    Args:
        experiment_dir: Directorio del experimento (ej: densenet121_cv_20250828_194941/)
        model_name: Nombre del modelo (ej: 'densenet121')
        num_classes: Número de clases del dataset
        device: Device donde cargar el modelo

    Returns:
        Modelo cargado y listo para inferencia
    """
    # Crear la arquitectura del modelo
    model = get_model(model_name, num_classes)

    # Cargar los pesos entrenados
    weights_path = experiment_dir / "final_model_weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontraron pesos en: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()  # Modo evaluación

    return model


def build_transforms(input_size: Tuple[int, int]):
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_stratified_splits(full_dataset, seed: int):
    """
    Crea splits balanceados:
    1. Test set (10%) - independiente, NO se toca
    2. Train+Val set (90%) - para k-fold CV

    Retorna índices estratificados por clase.
    """
    # Obtener todas las etiquetas
    all_labels = [full_dataset.samples[i][1] for i in range(len(full_dataset.samples))]
    all_indices = list(range(len(full_dataset.samples)))

    # Agrupar por clase para split estratificado manual
    class_indices = {}
    for idx, label in zip(all_indices, all_labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    test_indices = []
    trainval_indices = []

    set_seed(seed)
    for label, indices in class_indices.items():
        n_class = len(indices)
        n_test_class = max(1, round(TEST_SIZE * n_class))

        # Shuffle indices de esta clase
        indices = torch.tensor(indices)[torch.randperm(n_class)].tolist()

        test_indices.extend(indices[:n_test_class])
        trainval_indices.extend(indices[n_test_class:])

    # Shuffle los conjuntos finales
    test_indices = torch.tensor(test_indices)[torch.randperm(len(test_indices))].tolist()
    trainval_indices = torch.tensor(trainval_indices)[
        torch.randperm(len(trainval_indices))
    ].tolist()

    # Crear labels para train+val (para StratifiedKFold)
    trainval_labels = [all_labels[i] for i in trainval_indices]

    return test_indices, trainval_indices, trainval_labels


def train_epoch(model, dl, crit, opt, device):
    model.train()
    loss_sum = corr = 0
    for xb, yb in tqdm(dl, leave=False, desc="Training"):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * xb.size(0)
        corr += (out.argmax(1) == yb).sum().item()
    n = len(dl.dataset)
    return loss_sum / n, corr / n


def eval_epoch(model, dl, crit, device):
    model.eval()
    loss_sum = corr = 0
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in tqdm(dl, leave=False, desc="Evaluating"):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = crit(out, yb)
            loss_sum += loss.item() * xb.size(0)
            preds.append(out.argmax(1).cpu())
            labels.append(yb.cpu())
            corr += (out.argmax(1) == yb).sum().item()
    n = len(dl.dataset)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return loss_sum / n, corr / n, labels, preds


def plot_cv_curves(out_dir: Path, all_fold_histories: List[Dict]):
    """Plot training curves mostrando la variabilidad entre folds."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    metrics = ["train_loss", "train_acc", "val_loss", "val_acc"]
    titles = ["Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]

        # Calcular estadísticas por época
        max_epochs = max(len(fold_hist[metric]) for fold_hist in all_fold_histories)
        mean_values = []
        std_values = []

        for epoch in range(max_epochs):
            epoch_values = []
            for fold_hist in all_fold_histories:
                if epoch < len(fold_hist[metric]):
                    epoch_values.append(fold_hist[metric][epoch])

            if epoch_values:
                mean_values.append(np.mean(epoch_values))
                std_values.append(np.std(epoch_values))

        epochs = range(1, len(mean_values) + 1)
        mean_values = np.array(mean_values)
        std_values = np.array(std_values)

        # Plot con banda de confianza
        ax.plot(epochs, mean_values, "o-", label=f"Mean {metric}")
        ax.fill_between(
            epochs, mean_values - std_values, mean_values + std_values, alpha=0.3, label="±1 std"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} (K-Fold CV)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "cv_training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()


def save_aggregate_cm(
    out_dir: Path, all_y_true: List, all_y_pred: List, classes: List[str], tag: str
):
    """Crear matriz de confusión agregada de todos los folds."""
    # Concatenar todas las predicciones
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix Agregada - {tag} (K-Fold CV)")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_aggregate_{tag}.png", dpi=120, bbox_inches="tight")
    plt.close()


def calculate_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Calcula intervalo de confianza para una lista de valores."""
    if len(values) < 2:
        return 0.0, 0.0

    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of mean
    h = sem * stats.t.ppf((1 + confidence) / 2.0, len(values) - 1)
    return mean - h, mean + h


def compute_metrics_stats(metrics_list: List[float]) -> Dict[str, float]:
    """Calcula estadísticas robustas para una métrica."""
    if not metrics_list:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    mean_val = np.mean(metrics_list)
    std_val = np.std(metrics_list, ddof=1) if len(metrics_list) > 1 else 0.0
    ci_low, ci_high = calculate_confidence_interval(metrics_list)

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "min": round(np.min(metrics_list), 4),
        "max": round(np.max(metrics_list), 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    }


# ───────────────────────── Experiment routine with CV ───────────────────────
def run_experiment_cv(name: str, params: Dict, seed: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = VALIDATION_DIR / f"{name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device, dev_str = get_device()
    print(f"[✱] Using {dev_str}")
    set_seed(seed)

    # Parámetros del experimento
    model_name = params["model"]
    lr = float(params["lr"])
    epochs = int(params.get("epochs", 15))
    batch_size = int(params.get("batch_size", 32))
    patience = int(params.get("patience", 4))
    k_folds = int(params.get("k_folds", 5))

    print(f"[✱] Ejecutando {k_folds}-fold Cross Validation")

    # Preparar datos
    inp_size = get_input_size(model_name)
    tfm = build_transforms(inp_size)
    full_ds = datasets.ImageFolder(DATA_DIR, transform=tfm)
    classes = full_ds.classes
    num_classes = len(classes)

    # Crear splits estratificados
    test_indices, trainval_indices, trainval_labels = create_stratified_splits(full_ds, seed)

    # Crear test set independiente
    test_ds = Subset(full_ds, test_indices)
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # Configurar K-Fold CV
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # Almacenar resultados de cada fold
    fold_results = []
    all_fold_histories = []
    all_val_y_true = []
    all_val_y_pred = []

    total_cv_time = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(trainval_indices, trainval_labels), 1):
        print(f"\n[{name}] ═══ FOLD {fold}/{k_folds} ═══")

        # Convertir índices relativos a absolutos
        train_abs_idx = [trainval_indices[i] for i in train_idx]
        val_abs_idx = [trainval_indices[i] for i in val_idx]

        # Crear datasets para este fold
        train_ds = Subset(full_ds, train_abs_idx)
        val_ds = Subset(full_ds, val_abs_idx)

        train_dl = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )

        print(f"    Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

        # Crear modelo fresco para este fold
        model = get_model(model_name, num_classes).to(device)
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=lr)

        # Historia de entrenamiento para este fold
        fold_hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
        best_val_acc = 0
        epochs_no_imp = 0
        best_model_state = None

        t0 = time.time()
        for ep in range(1, epochs + 1):
            tl, ta = train_epoch(model, train_dl, crit, opt, device)
            vl, va, _, _ = eval_epoch(model, val_dl, crit, device)

            fold_hist["train_loss"].append(tl)
            fold_hist["train_acc"].append(ta)
            fold_hist["val_loss"].append(vl)
            fold_hist["val_acc"].append(va)

            print(f"    Ep{ep:02d} TL={tl:.3f} TA={ta:.3f} VL={vl:.3f} VA={va:.3f}")

            if va > best_val_acc:
                best_val_acc = va
                epochs_no_imp = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= patience:
                    print(f"    Early stop en época {ep}")
                    break

        fold_train_time = time.time() - t0
        total_cv_time += fold_train_time

        # Cargar mejor modelo y evaluar
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        val_loss, val_acc, val_y_true, val_y_pred = eval_epoch(model, val_dl, crit, device)

        # Calcular métricas detalladas
        val_p, val_r, val_f1, _ = precision_recall_fscore_support(
            val_y_true, val_y_pred, average="macro", zero_division=0
        )

        # Calcular datos para intervalos de confianza Wilson
        val_n = len(val_y_true)
        val_k = int((np.array(val_y_true) == np.array(val_y_pred)).sum())

        fold_result = {
            "fold": fold,
            "train_time_sec": round(fold_train_time, 2),
            "final_epoch": ep,
            "best_val_acc": round(best_val_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_precision": round(val_p, 4),
            "val_recall": round(val_r, 4),
            "val_f1": round(val_f1, 4),
            "val_samples": val_n,
            "val_correct": val_k,
        }

        fold_results.append(fold_result)
        all_fold_histories.append(fold_hist)
        all_val_y_true.append(val_y_true)
        all_val_y_pred.append(val_y_pred)

        print(f"    ✓ Fold {fold} completado: Val Acc={val_acc:.3f}, F1={val_f1:.3f}")

    print(f"\n[{name}] ═══ EVALUACIÓN FINAL EN TEST SET ═══")

    # Entrenar modelo final con TODOS los datos de train+val para test final
    trainval_ds = Subset(full_ds, trainval_indices)
    trainval_dl = DataLoader(
        trainval_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )

    final_model = get_model(model_name, num_classes).to(device)
    final_crit = nn.CrossEntropyLoss()
    final_opt = optim.Adam(final_model.parameters(), lr=lr)

    # Entrenar modelo final (usando épocas promedio de CV)
    avg_epochs = int(np.mean([fr["final_epoch"] for fr in fold_results]))
    print(f"    Entrenando modelo final por {avg_epochs} épocas...")

    for ep in range(1, avg_epochs + 1):
        tl, ta = train_epoch(final_model, trainval_dl, final_crit, final_opt, device)
        if ep % 5 == 0:
            print(f"    Ep{ep:02d} TL={tl:.3f} TA={ta:.3f}")

    # Evaluación final en test set
    test_loss, test_acc, test_y_true, test_y_pred = eval_epoch(
        final_model, test_dl, final_crit, device
    )
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(
        test_y_true, test_y_pred, average="macro", zero_division=0
    )

    print(f"    ✓ Test final: Acc={test_acc:.3f}, F1={test_f1:.3f}")

    # Guardar modelo final entrenado con todos los datos train+val
    torch.save(final_model.state_dict(), out_dir / "final_model_weights.pth")
    print(f"    ✓ Modelo final guardado: final_model_weights.pth")

    # ═══════════════════════════════════════════════════════════════════════════
    # ANÁLISIS ESTADÍSTICO Y GUARDADO DE RESULTADOS
    # ═══════════════════════════════════════════════════════════════════════════

    # Extraer métricas para análisis estadístico
    val_accs = [fr["val_acc"] for fr in fold_results]
    val_f1s = [fr["val_f1"] for fr in fold_results]
    val_precisions = [fr["val_precision"] for fr in fold_results]
    val_recalls = [fr["val_recall"] for fr in fold_results]
    val_losses = [fr["val_loss"] for fr in fold_results]

    # Calcular estadísticas robustas
    cv_stats = {
        "val_acc": compute_metrics_stats(val_accs),
        "val_f1": compute_metrics_stats(val_f1s),
        "val_precision": compute_metrics_stats(val_precisions),
        "val_recall": compute_metrics_stats(val_recalls),
        "val_loss": compute_metrics_stats(val_losses),
    }

    # Guardar resultados detallados
    detailed_results = {
        "experiment_name": name,
        "model": model_name,
        "hyperparameters": {
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "k_folds": k_folds,
        },
        "cv_configuration": {
            "test_size": TEST_SIZE,
            "train_val_size": TRAIN_VAL_SIZE,
            "total_samples": len(full_ds),
            "test_samples": len(test_ds),
            "trainval_samples": len(trainval_indices),
        },
        "fold_results": fold_results,
        "cv_statistics": cv_stats,
        "test_results": {
            "test_acc": round(test_acc, 4),
            "test_precision": round(test_p, 4),
            "test_recall": round(test_r, 4),
            "test_f1": round(test_f1, 4),
            "test_loss": round(test_loss, 4),
        },
        "metadata": {
            "timestamp": ts,
            "device": dev_str,
            "seed": seed,
            "total_cv_time_sec": round(total_cv_time, 2),
            "avg_fold_time_sec": round(total_cv_time / k_folds, 2),
            "final_model_saved": "final_model_weights.pth",
            "final_model_trained_epochs": avg_epochs,
        },
    }

    # Guardar archivo JSON detallado
    (out_dir / "detailed_results.json").write_text(json.dumps(detailed_results, indent=2))

    # Guardar resultados por fold
    (out_dir / "fold_results.json").write_text(json.dumps(fold_results, indent=2))

    # Crear visualizaciones
    plot_cv_curves(out_dir, all_fold_histories)
    save_aggregate_cm(out_dir, all_val_y_true, all_val_y_pred, classes, "validation")
    save_aggregate_cm(out_dir, [test_y_true], [test_y_pred], classes, "test")

    # Crear resumen para CSV
    summary_row = {
        "name": name,
        "model": model_name,
        "k_folds": k_folds,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "device": dev_str,
        # Métricas de CV (con intervalos de confianza)
        "cv_val_acc_mean": cv_stats["val_acc"]["mean"],
        "cv_val_acc_std": cv_stats["val_acc"]["std"],
        "cv_val_acc_ci_low": cv_stats["val_acc"]["ci_low"],
        "cv_val_acc_ci_high": cv_stats["val_acc"]["ci_high"],
        "cv_val_f1_mean": cv_stats["val_f1"]["mean"],
        "cv_val_f1_std": cv_stats["val_f1"]["std"],
        "cv_val_f1_ci_low": cv_stats["val_f1"]["ci_low"],
        "cv_val_f1_ci_high": cv_stats["val_f1"]["ci_high"],
        "cv_val_precision_mean": cv_stats["val_precision"]["mean"],
        "cv_val_precision_std": cv_stats["val_precision"]["std"],
        "cv_val_recall_mean": cv_stats["val_recall"]["mean"],
        "cv_val_recall_std": cv_stats["val_recall"]["std"],
        # Métricas de test final
        "test_acc": round(test_acc, 4),
        "test_precision": round(test_p, 4),
        "test_recall": round(test_r, 4),
        "test_f1": round(test_f1, 4),
        "test_loss": round(test_loss, 4),
        # Metadatos
        "total_cv_time_sec": round(total_cv_time, 2),
        "avg_fold_time_sec": round(total_cv_time / k_folds, 2),
        "timestamp": ts,
    }

    # Guardar en CSV
    summary_path = VALIDATION_DIR / "summary_cv.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header:
            csv_writer.writeheader()
        csv_writer.writerow(summary_row)

    # Imprimir resumen final
    print(f"\n{'='*80}")
    print(f"RESULTADOS FINALES - {name}")
    print(f"{'='*80}")
    print(f"Validación Cruzada ({k_folds}-fold):")

    # Truncar intervalos de confianza a [0,1] para métricas de rendimiento
    acc_ci_low = max(0.0, min(1.0, cv_stats["val_acc"]["ci_low"]))
    acc_ci_high = max(0.0, min(1.0, cv_stats["val_acc"]["ci_high"]))
    f1_ci_low = max(0.0, min(1.0, cv_stats["val_f1"]["ci_low"]))
    f1_ci_high = max(0.0, min(1.0, cv_stats["val_f1"]["ci_high"]))

    # Calcular intervalo de confianza Wilson para accuracy (más apropiado para proporciones)
    total_samples = sum(fr["val_samples"] for fr in fold_results)
    total_correct = sum(fr["val_correct"] for fr in fold_results)
    if total_samples > 0:
        from scipy import stats

        z = stats.norm.ppf(0.975)  # 95% CI
        p_hat = total_correct / total_samples
        denom = 1 + z**2 / total_samples
        center = (p_hat + z**2 / (2 * total_samples)) / denom
        halfwidth = (
            z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total_samples)) / total_samples)
        ) / denom
        wilson_low = max(0.0, center - halfwidth)
        wilson_high = min(1.0, center + halfwidth)

        print(
            f"  • Accuracy:  {cv_stats['val_acc']['mean']:.3f} ± {cv_stats['val_acc']['std']:.3f} "
            f"IC-t:[{acc_ci_low:.3f}, {acc_ci_high:.3f}] Wilson:[{wilson_low:.3f}, {wilson_high:.3f}]"
        )
    else:
        print(
            f"  • Accuracy:  {cv_stats['val_acc']['mean']:.3f} ± {cv_stats['val_acc']['std']:.3f} "
            f"[{acc_ci_low:.3f}, {acc_ci_high:.3f}]"
        )
    print(
        f"  • F1-Score:  {cv_stats['val_f1']['mean']:.3f} ± {cv_stats['val_f1']['std']:.3f} "
        f"[{f1_ci_low:.3f}, {f1_ci_high:.3f}]"
    )
    print(
        f"  • Precision: {cv_stats['val_precision']['mean']:.3f} ± {cv_stats['val_precision']['std']:.3f}"
    )
    print(
        f"  • Recall:    {cv_stats['val_recall']['mean']:.3f} ± {cv_stats['val_recall']['std']:.3f}"
    )
    print(f"\nTest Set Final:")
    print(f"  • Accuracy:  {test_acc:.3f}")
    print(f"  • F1-Score:  {test_f1:.3f}")
    print(f"  • Precision: {test_p:.3f}")
    print(f"  • Recall:    {test_r:.3f}")
    print(f"\nTiempo total: {total_cv_time:.1f}s (promedio por fold: {total_cv_time/k_folds:.1f}s)")
    print(f"Resultados guardados en: {out_dir}")


# ─────────────────────────────── main ───────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with K-Fold Cross Validation")
    parser.add_argument("--config", type=Path, default=EXPERIMENTS_PATH, help="YAML config path")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file {args.config} not found!")
        print("Creating example config file...")

        example_config = {
            "experiments": [
                {
                    "name": "resnet50_cv_example",
                    "execute": True,
                    "params": {
                        "model": "resnet50",
                        "lr": 1e-4,
                        "epochs": 10,
                        "batch_size": 32,
                        "patience": 3,
                        "k_folds": 5,
                    },
                }
            ]
        }

        args.config.write_text(yaml.dump(example_config, default_flow_style=False))
        print(f"Example config created at: {args.config}")
        print("Edit the file and run again.")
        exit(1)

    cfg = yaml.safe_load(args.config.read_text())
    exps = cfg.get("experiments", [])
    if not exps:
        raise ValueError("No experiments defined in YAML")

    # Filtrar solo los experimentos que deben ejecutarse
    exps_to_run = [exp for exp in exps if exp.get("execute", True)]
    total_exps = len(exps_to_run)
    print(f"Running {total_exps} experiment(s) with Cross Validation out of {len(exps)} defined…")

    for i, exp in enumerate(exps_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT {i}/{total_exps}: {exp['name']}")
        print(f"{'#'*80}")
        run_experiment_cv(exp["name"], exp["params"], seed=42)

    print(f"\n{'='*80}")
    print(f"✓ ALL {total_exps} EXPERIMENTS COMPLETED")
    print(f"✓ Results saved in: {VALIDATION_DIR}")
    print(f"✓ Summary CSV: {VALIDATION_DIR / 'summary_cv.csv'}")
    print(f"{'='*80}")
