#!/usr/bin/env python3
"""
Analizador de resultados de validación cruzada.

Este script lee los resultados de experiments_cv.py y genera:
• Comparaciones estadísticas entre modelos
• Gráficos de barras con intervalos de confianza
• Análisis de significancia estadística
• Tablas resumen formateadas para papers

Usage:
    python analyze_cv_results.py
"""

import json
import warnings
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Configurar estilo de plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

PROJECT_ROOT = Path(
    "/home/nahumfg/GithubProjects/parliament-voting-records/validation/classification"
)
VALIDATION_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results() -> List[Dict]:
    """Carga todos los resultados detallados de experimentos."""
    results = []

    for result_dir in VALIDATION_DIR.glob("*_2*"):  # Buscar directorios con timestamp
        detailed_file = result_dir / "detailed_results.json"
        if detailed_file.exists():
            with open(detailed_file, "r") as f:
                data = json.load(f)
                results.append(data)

    return results


def create_summary_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Crea DataFrame resumen con estadísticas de CV."""
    summary_data = []

    for result in results:
        exp_name = result["experiment_name"]
        model = result["model"]
        cv_stats = result["cv_statistics"]
        test_results = result["test_results"]
        metadata = result["metadata"]

        row = {
            "experiment": exp_name,
            "model": model,
            "k_folds": result["hyperparameters"]["k_folds"],
            # Métricas de CV
            "cv_acc_mean": cv_stats["val_acc"]["mean"],
            "cv_acc_std": cv_stats["val_acc"]["std"],
            "cv_acc_ci_low": cv_stats["val_acc"]["ci_low"],
            "cv_acc_ci_high": cv_stats["val_acc"]["ci_high"],
            "cv_f1_mean": cv_stats["val_f1"]["mean"],
            "cv_f1_std": cv_stats["val_f1"]["std"],
            "cv_f1_ci_low": cv_stats["val_f1"]["ci_low"],
            "cv_f1_ci_high": cv_stats["val_f1"]["ci_high"],
            "cv_precision_mean": cv_stats["val_precision"]["mean"],
            "cv_precision_std": cv_stats["val_precision"]["std"],
            "cv_recall_mean": cv_stats["val_recall"]["mean"],
            "cv_recall_std": cv_stats["val_recall"]["std"],
            # Métricas de test
            "test_acc": test_results["test_acc"],
            "test_f1": test_results["test_f1"],
            "test_precision": test_results["test_precision"],
            "test_recall": test_results["test_recall"],
            # Metadatos
            "total_time_sec": metadata["total_cv_time_sec"],
            "avg_fold_time_sec": metadata["avg_fold_time_sec"],
            "timestamp": metadata["timestamp"],
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Clipear intervalos de confianza a [0,1]
    df = _clip_ci_cols(df, ["cv_acc_ci_low", "cv_acc_ci_high", "cv_f1_ci_low", "cv_f1_ci_high"])

    # Añadir intervalos Wilson para accuracy si están disponibles los datos
    df = add_wilson_to_df(df, results)

    return df


def plot_cv_comparison(df: pd.DataFrame):
    """Crear gráfico de comparación con intervalos de confianza."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics = ["cv_acc", "cv_f1", "cv_precision", "cv_recall"]
    titles = ["Accuracy", "F1-Score", "Precision", "Recall"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        x_pos = np.arange(len(df))
        means = df[f"{metric}_mean"].values

        # Usar IC como barras de error si están disponibles, sino usar std
        if f"{metric}_ci_low" in df.columns and f"{metric}_ci_high" in df.columns:
            ci_low = df[f"{metric}_ci_low"].values
            ci_high = df[f"{metric}_ci_high"].values
            error_bars = [means - ci_low, ci_high - means]
        else:
            stds = df[f"{metric}_std"].values
            error_bars = stds

        # Barras con error bars
        bars = ax.bar(
            x_pos,
            means,
            yerr=error_bars,
            capsize=5,
            alpha=0.7,
            color=sns.color_palette("husl", len(df)),
        )

        # Intervalos de confianza como líneas horizontales
        if f"{metric}_ci_low" in df.columns and f"{metric}_ci_high" in df.columns:
            ci_low = df[f"{metric}_ci_low"].values
            ci_high = df[f"{metric}_ci_high"].values

            for i, (low, high) in enumerate(zip(ci_low, ci_high)):
                ax.plot([i - 0.2, i + 0.2], [low, low], "k-", linewidth=2)
                ax.plot([i - 0.2, i + 0.2], [high, high], "k-", linewidth=2)
                ax.plot([i, i], [low, high], "k-", linewidth=1)

        ax.set_xlabel("Experimento")
        ax.set_ylabel(title)
        ax.set_title(f"{title} - Validación Cruzada")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["experiment"].values, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Fijar ylim para métricas [0,1]
        if metric in ["cv_acc", "cv_f1", "cv_precision", "cv_recall"]:
            ax.set_ylim(0.0, 1.0)

        # Añadir valores en las barras
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            if f"{metric}_ci_low" in df.columns and f"{metric}_ci_high" in df.columns:
                ci_low = df[f"{metric}_ci_low"].iloc[i]
                ci_high = df[f"{metric}_ci_high"].iloc[i]
                text_y = min(ci_high + 0.01, 0.95)  # No exceder ylim
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    text_y,
                    f"{mean:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            else:
                std = df[f"{metric}_std"].iloc[i]
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std + 0.01,
                    f"{mean:.3f}±{std:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Gráfico de comparación guardado en: {OUTPUT_DIR}/cv_comparison.png")


def plot_cv_vs_test(df: pd.DataFrame):
    """Comparar performance de CV vs Test."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy
    ax1.scatter(df["cv_acc_mean"], df["test_acc"], s=100, alpha=0.7)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)  # Línea diagonal

    for i, experiment in enumerate(df["experiment"]):
        ax1.annotate(
            experiment,
            (df["cv_acc_mean"].iloc[i], df["test_acc"].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax1.set_xlabel("CV Accuracy (mean)")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("CV vs Test Accuracy")
    ax1.grid(True, alpha=0.3)

    # F1-Score
    ax2.scatter(df["cv_f1_mean"], df["test_f1"], s=100, alpha=0.7)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)  # Línea diagonal

    for i, experiment in enumerate(df["experiment"]):
        ax2.annotate(
            experiment,
            (df["cv_f1_mean"].iloc[i], df["test_f1"].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax2.set_xlabel("CV F1-Score (mean)")
    ax2.set_ylabel("Test F1-Score")
    ax2.set_title("CV vs Test F1-Score")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_vs_test.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Gráfico CV vs Test guardado en: {OUTPUT_DIR}/cv_vs_test.png")


def plot_confidence_intervals(df: pd.DataFrame):
    """Crear gráfico específico de intervalos de confianza."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    experiments = df["experiment"].tolist()
    x_pos = np.arange(len(experiments))

    # Accuracy con IC 95%
    acc_means = df["cv_acc_mean"].values
    acc_ci_low = df["cv_acc_ci_low"].values
    acc_ci_high = df["cv_acc_ci_high"].values
    acc_errors = [acc_means - acc_ci_low, acc_ci_high - acc_means]

    bars1 = ax1.bar(x_pos, acc_means, alpha=0.7, color=sns.color_palette("husl", len(df)))
    ax1.errorbar(
        x_pos,
        acc_means,
        yerr=acc_errors,
        fmt="none",
        color="black",
        capsize=5,
        capthick=2,
        linewidth=2,
        label="IC 95%",
    )

    ax1.set_xlabel("Experimento")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy con Intervalos de Confianza (95%)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(experiments, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0.0, 1.0)

    # Agregar valores sobre las barras
    for i, (bar, mean, ci_low, ci_high) in enumerate(
        zip(bars1, acc_means, acc_ci_low, acc_ci_high)
    ):
        height = bar.get_height()
        text_y = min(ci_high + 0.005, 0.95)
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y,
            f"{mean:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # F1-Score con IC 95%
    f1_means = df["cv_f1_mean"].values
    f1_ci_low = df["cv_f1_ci_low"].values
    f1_ci_high = df["cv_f1_ci_high"].values
    f1_errors = [f1_means - f1_ci_low, f1_ci_high - f1_means]

    bars2 = ax2.bar(x_pos, f1_means, alpha=0.7, color=sns.color_palette("husl", len(df)))
    ax2.errorbar(
        x_pos,
        f1_means,
        yerr=f1_errors,
        fmt="none",
        color="black",
        capsize=5,
        capthick=2,
        linewidth=2,
        label="IC 95%",
    )

    ax2.set_xlabel("Experimento")
    ax2.set_ylabel("F1-Score")
    ax2.set_title("F1-Score con Intervalos de Confianza (95%)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(experiments, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0.0, 1.0)

    # Agregar valores sobre las barras
    for i, (bar, mean, ci_low, ci_high) in enumerate(zip(bars2, f1_means, f1_ci_low, f1_ci_high)):
        height = bar.get_height()
        text_y = min(ci_high + 0.005, 0.95)
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y,
            f"{mean:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_intervals.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(
        f"✓ Gráfico de intervalos de confianza guardado en: {OUTPUT_DIR}/confidence_intervals.png"
    )


def cohens_d_paired(a, b):
    """Calcular Cohen's d para muestras pareadas."""
    a = np.array(a)
    b = np.array(b)
    diff = a - b
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def cliffs_delta(a, b):
    """Tamaño de efecto no paramétrico (robusto a techo/piso)."""
    a = np.array(a)
    b = np.array(b)
    greater = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    return float((greater - less) / (len(a) * len(b) + 1e-12))


def holm_bonferroni(pvals, alpha=0.05):
    """Corrección Holm-Bonferroni para comparaciones múltiples."""
    m = len(pvals)
    if m == 0:
        return [], []

    order = np.argsort(pvals)
    adjusted = [None] * m
    sig = [False] * m

    for k, idx in enumerate(order, 1):
        thr = alpha / (m - k + 1)
        sig[idx] = pvals[idx] < thr
        adjusted[idx] = min(pvals[idx] * (m - k + 1), 1.0)

    return adjusted, sig


def wilson_ci(k, n, conf=0.95):
    """Intervalos de confianza Wilson para proporciones."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(0.5 + conf / 2.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    halfwidth = (z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    lo = max(0.0, center - halfwidth)
    hi = min(1.0, center + halfwidth)
    return float(lo), float(hi)


def _clip_ci_cols(df, cols):
    """Clipear columnas de intervalos de confianza al rango [0,1]."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0.0, upper=1.0)
    return df


def add_wilson_to_df(df, results):
    """Añadir intervalos de confianza Wilson para accuracy global por experimento."""
    lo_list, hi_list = [], []
    for r in results:
        k = sum(fr.get("val_correct", 0) for fr in r["fold_results"])
        n = sum(fr.get("val_samples", 0) for fr in r["fold_results"])
        if n > 0:
            lo, hi = wilson_ci(k, n, conf=0.95)
        else:
            lo, hi = 0.0, 0.0
        lo_list.append(lo)
        hi_list.append(hi)

    df["cv_acc_wilson_low"] = lo_list
    df["cv_acc_wilson_high"] = hi_list
    return df


def analyze_confidence_intervals(df: pd.DataFrame):
    """Análisis de solapamiento de intervalos de confianza para significancia."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE INTERVALOS DE CONFIANZA (IC 95%)")
    print("=" * 80)

    # Análisis de solapamiento para Accuracy
    print("\n📊 ACCURACY - Intervalos de Confianza:")
    print("-" * 50)

    for i, row in df.iterrows():
        experiment = row["experiment"]
        model = row["model"]
        mean_acc = row["cv_acc_mean"]
        ci_low = row["cv_acc_ci_low"]
        ci_high = row["cv_acc_ci_high"]
        width = ci_high - ci_low

        print(
            f"{experiment:20s} ({model:12s}): {mean_acc:.3f} [{ci_low:.3f}, {ci_high:.3f}] (ancho: {width:.3f})"
        )

    # Mostrar Wilson CI si está disponible
    if "cv_acc_wilson_low" in df.columns:
        print("\n📊 ACCURACY - Intervalos Wilson (95%):")
        print("-" * 50)
        for i, row in df.iterrows():
            experiment = row["experiment"]
            model = row["model"]
            mean_acc = row["cv_acc_mean"]
            wilson_low = row["cv_acc_wilson_low"]
            wilson_high = row["cv_acc_wilson_high"]
            width = wilson_high - wilson_low

            print(
                f"{experiment:20s} ({model:12s}): {mean_acc:.3f} [{wilson_low:.3f}, {wilson_high:.3f}] (ancho: {width:.3f})"
            )

    # Análisis de solapamiento para F1
    print("\n📊 F1-SCORE - Intervalos de Confianza:")
    print("-" * 50)

    for i, row in df.iterrows():
        experiment = row["experiment"]
        model = row["model"]
        mean_f1 = row["cv_f1_mean"]
        ci_low = row["cv_f1_ci_low"]
        ci_high = row["cv_f1_ci_high"]
        width = ci_high - ci_low

        print(
            f"{experiment:20s} ({model:12s}): {mean_f1:.3f} [{ci_low:.3f}, {ci_high:.3f}] (ancho: {width:.3f})"
        )

    # Análisis de significancia por solapamiento
    print("\n🔍 ANÁLISIS DE SIGNIFICANCIA (no solapamiento de IC):")
    print("-" * 60)

    experiments = df["experiment"].tolist()
    models = df["model"].tolist()
    significant_pairs = []

    for i in range(len(experiments)):
        for j in range(i + 1, len(experiments)):
            exp1, exp2 = experiments[i], experiments[j]
            model1, model2 = models[i], models[j]

            # Accuracy
            acc1_low = df.iloc[i]["cv_acc_ci_low"]
            acc1_high = df.iloc[i]["cv_acc_ci_high"]
            acc2_low = df.iloc[j]["cv_acc_ci_low"]
            acc2_high = df.iloc[j]["cv_acc_ci_high"]

            acc_no_overlap = (acc1_high < acc2_low) or (acc2_high < acc1_low)

            # F1
            f1_1_low = df.iloc[i]["cv_f1_ci_low"]
            f1_1_high = df.iloc[i]["cv_f1_ci_high"]
            f1_2_low = df.iloc[j]["cv_f1_ci_low"]
            f1_2_high = df.iloc[j]["cv_f1_ci_high"]

            f1_no_overlap = (f1_1_high < f1_2_low) or (f1_2_high < f1_1_low)

            if acc_no_overlap or f1_no_overlap:
                better_exp = exp1 if df.iloc[i]["cv_acc_mean"] > df.iloc[j]["cv_acc_mean"] else exp2
                worse_exp = exp2 if better_exp == exp1 else exp1
                better_model = model1 if better_exp == exp1 else model2
                worse_model = model2 if better_exp == exp1 else model1
                metric = "Accuracy" if acc_no_overlap else "F1-Score"
                significant_pairs.append((better_exp, worse_exp, metric))
                print(
                    f"✅ {better_exp} ({better_model}) > {worse_exp} ({worse_model}) ({metric}) - Diferencia significativa"
                )

    if not significant_pairs:
        print("⚠️  Ningún par muestra diferencias estadísticamente significativas")
        print("   (todos los intervalos de confianza se solapan)")

    print("\n⚠️  Nota: El solapamiento/no-solapamiento de IC es solo indicativo.")
    print("    La significancia formal se determina con pruebas pareadas (t-test/Wilcoxon)")
    print("    y corrección por multiplicidad (Holm–Bonferroni).")


def statistical_comparison(df: pd.DataFrame, results: List[Dict]):
    """Realizar comparaciones estadísticas pareadas entre modelos."""
    print("\n" + "=" * 80)
    print("ANÁLISIS ESTADÍSTICO PAREADO DE RESULTADOS")
    print("=" * 80)

    # Obtener datos por fold para cada experimento
    fold_data = {}
    for result in results:
        exp_name = result["experiment_name"]
        fold_results = result["fold_results"]
        fold_data[exp_name] = {
            "accuracies": [fr["val_acc"] for fr in fold_results],
            "f1_scores": [fr["val_f1"] for fr in fold_results],
            "model": result["model"],
            "experiment": exp_name,
        }

    # Comparar todos los pares de experimentos
    experiments = list(fold_data.keys())

    # Armar las comparaciones con tests pareados
    pair_labels = []
    pvals_acc = []
    pvals_f1 = []
    results_rows = []

    for i in range(len(experiments)):
        for j in range(i + 1, len(experiments)):
            exp1, exp2 = experiments[i], experiments[j]
            acc1 = fold_data[exp1]["accuracies"]
            acc2 = fold_data[exp2]["accuracies"]
            f1_1 = fold_data[exp1]["f1_scores"]
            f1_2 = fold_data[exp2]["f1_scores"]

            # Tests pareados
            t_stat_acc, p_acc = stats.ttest_rel(acc1, acc2)
            t_stat_f1, p_f1 = stats.ttest_rel(f1_1, f1_2)

            # Alternativa no paramétrica (robusto)
            try:
                w_acc, p_wacc = stats.wilcoxon(acc1, acc2, zero_method="zsplit", correction=True)
                w_f1, p_wf1 = stats.wilcoxon(f1_1, f1_2, zero_method="zsplit", correction=True)
            except ValueError:
                # Si todas las diferencias son cero
                w_acc, p_wacc = 0, 1.0
                w_f1, p_wf1 = 0, 1.0

            # Tamaños de efecto
            d_acc = cohens_d_paired(acc1, acc2)
            d_f1 = cohens_d_paired(f1_1, f1_2)
            cd_acc = cliffs_delta(acc1, acc2)
            cd_f1 = cliffs_delta(f1_1, f1_2)

            # Guarda p-values para corrección por multiplicidad
            pair_labels.append(f"{exp1} vs {exp2}")
            pvals_acc.append(p_acc)
            pvals_f1.append(p_f1)

            results_rows.append(
                {
                    "pair": f"{exp1} vs {exp2}",
                    "t_acc": t_stat_acc,
                    "p_acc": p_acc,
                    "wilcoxon_acc_p": p_wacc,
                    "t_f1": t_stat_f1,
                    "p_f1": p_f1,
                    "wilcoxon_f1_p": p_wf1,
                    "cohens_d_acc": d_acc,
                    "cohens_d_f1": d_f1,
                    "cliffs_delta_acc": cd_acc,
                    "cliffs_delta_f1": cd_f1,
                }
            )

    # Corrección Holm–Bonferroni (separada por métrica)
    adj_acc, sig_acc = holm_bonferroni(pvals_acc, alpha=0.05)
    adj_f1, sig_f1 = holm_bonferroni(pvals_f1, alpha=0.05)

    print("\nComparaciones pareadas con corrección Holm–Bonferroni:")
    print("-" * 80)

    for k, row in enumerate(results_rows):
        row["p_acc_holm"] = adj_acc[k]
        row["sig_acc_holm"] = sig_acc[k]
        row["p_f1_holm"] = adj_f1[k]
        row["sig_f1_holm"] = sig_f1[k]

        print(f"- {row['pair']}:")
        print(
            f"  ACC: t={row['t_acc']:.3f} p={row['p_acc']:.4g} (holm={row['p_acc_holm']:.4g}, sig={row['sig_acc_holm']})"
        )
        print(
            f"  F1:  t={row['t_f1']:.3f} p={row['p_f1']:.4g} (holm={row['p_f1_holm']:.4g}, sig={row['sig_f1_holm']})"
        )
        print(f"  Wilcoxon: ACC p={row['wilcoxon_acc_p']:.4g}, F1 p={row['wilcoxon_f1_p']:.4g}")
        print(f"  Tamaños efecto: d_acc={row['cohens_d_acc']:.3f}, d_f1={row['cohens_d_f1']:.3f}")
        print(
            f"                  δ_acc={row['cliffs_delta_acc']:.3f}, δ_f1={row['cliffs_delta_f1']:.3f}"
        )
        print()

    print("📝 INTERPRETACIÓN:")
    print("  • t-test pareado + corrección Holm: test formal de significancia")
    print("  • Wilcoxon: alternativa no paramétrica (robusta a métricas en techo)")
    print("  • Cohen's d: tamaño de efecto paramétrico (|d|>0.5 = mediano, |d|>0.8 = grande)")
    print(
        "  • Cliff's delta: tamaño de efecto no paramétrico (|δ|>0.28 = mediano, |δ|>0.43 = grande)"
    )


def generate_latex_table(df: pd.DataFrame):
    """Generar tabla en formato LaTeX para papers."""
    latex_table = """
\\begin{table}[h]
\\centering
\\caption{Resultados de Validación Cruzada para Clasificación de Imágenes}
\\label{tab:cv_results}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Experimento} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} \\\\
\\midrule
"""

    for _, row in df.iterrows():
        experiment = row["experiment"].replace("_", "\\_")
        model = row["model"].replace("_", "\\_")
        acc_mean = row["cv_acc_mean"]
        acc_std = row["cv_acc_std"]
        f1_mean = row["cv_f1_mean"]
        f1_std = row["cv_f1_std"]
        prec_mean = row["cv_precision_mean"]
        prec_std = row["cv_precision_std"]
        rec_mean = row["cv_recall_mean"]
        rec_std = row["cv_recall_std"]

        # Obtener intervalos de confianza
        acc_ci_low = row["cv_acc_ci_low"]
        acc_ci_high = row["cv_acc_ci_high"]
        f1_ci_low = row["cv_f1_ci_low"]
        f1_ci_high = row["cv_f1_ci_high"]

        latex_table += f"{experiment} & {acc_mean:.3f} ± {acc_std:.3f} & {f1_mean:.3f} ± {f1_std:.3f} & {prec_mean:.3f} ± {prec_std:.3f} & {rec_mean:.3f} ± {rec_std:.3f} \\\\\n"

        # Añadir Wilson CI si está disponible
        if "cv_acc_wilson_low" in row.index:
            wilson_low = row["cv_acc_wilson_low"]
            wilson_high = row["cv_acc_wilson_high"]
            latex_table += f"({model}) & Wilson: [{wilson_low:.3f}, {wilson_high:.3f}], IC-t: [{acc_ci_low:.3f}, {acc_ci_high:.3f}] & [{f1_ci_low:.3f}, {f1_ci_high:.3f}] & & \\\\\n"
        else:
            latex_table += f"({model}) & [{acc_ci_low:.3f}, {acc_ci_high:.3f}] & [{f1_ci_low:.3f}, {f1_ci_high:.3f}] & & \\\\\n"

    latex_table += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Valores reportados como media ± desviación estándar de validación cruzada 5-fold.
\\item Todas las métricas calculadas usando promedio macro.
\\end{tablenotes}
\\end{table}
"""

    # Guardar tabla LaTeX
    with open(OUTPUT_DIR / "results_table.tex", "w") as f:
        f.write(latex_table)

    print(f"✓ Tabla LaTeX guardada en: {OUTPUT_DIR}/results_table.tex")


def create_summary_report(df: pd.DataFrame):
    """Crear reporte resumen en texto."""
    report = f"""
RESUMEN EJECUTIVO - VALIDACIÓN CRUZADA
{'='*50}

Número de experimentos: {len(df)}
Modelos evaluados: {', '.join(df['model'].unique())}

MEJORES RESULTADOS:
{'-'*30}

Mejor Accuracy (CV):
{df.loc[df['cv_acc_mean'].idxmax(), 'experiment']} ({df.loc[df['cv_acc_mean'].idxmax(), 'model']}) - {df['cv_acc_mean'].max():.3f} ± {df.loc[df['cv_acc_mean'].idxmax(), 'cv_acc_std']:.3f}
IC 95%: [{df.loc[df['cv_acc_mean'].idxmax(), 'cv_acc_ci_low']:.3f}, {df.loc[df['cv_acc_mean'].idxmax(), 'cv_acc_ci_high']:.3f}]

Mejor F1-Score (CV):
{df.loc[df['cv_f1_mean'].idxmax(), 'experiment']} ({df.loc[df['cv_f1_mean'].idxmax(), 'model']}) - {df['cv_f1_mean'].max():.3f} ± {df.loc[df['cv_f1_mean'].idxmax(), 'cv_f1_std']:.3f}
IC 95%: [{df.loc[df['cv_f1_mean'].idxmax(), 'cv_f1_ci_low']:.3f}, {df.loc[df['cv_f1_mean'].idxmax(), 'cv_f1_ci_high']:.3f}]

Mejor Accuracy (Test):
{df.loc[df['test_acc'].idxmax(), 'experiment']} ({df.loc[df['test_acc'].idxmax(), 'model']}) - {df['test_acc'].max():.3f}

Mejor F1-Score (Test):
{df.loc[df['test_f1'].idxmax(), 'experiment']} ({df.loc[df['test_f1'].idxmax(), 'model']}) - {df['test_f1'].max():.3f}

ANÁLISIS DE VARIABILIDAD:
{'-'*30}

Experimento más consistente (menor std en accuracy):
{df.loc[df['cv_acc_std'].idxmin(), 'experiment']} ({df.loc[df['cv_acc_std'].idxmin(), 'model']}) - std = {df['cv_acc_std'].min():.4f}

Experimento menos consistente (mayor std en accuracy):
{df.loc[df['cv_acc_std'].idxmax(), 'experiment']} ({df.loc[df['cv_acc_std'].idxmax(), 'model']}) - std = {df['cv_acc_std'].max():.4f}

EFICIENCIA TEMPORAL:
{'-'*30}

Experimento más rápido:
{df.loc[df['avg_fold_time_sec'].idxmin(), 'experiment']} ({df.loc[df['avg_fold_time_sec'].idxmin(), 'model']}) - {df['avg_fold_time_sec'].min():.1f}s por fold

Experimento más lento:
{df.loc[df['avg_fold_time_sec'].idxmax(), 'experiment']} ({df.loc[df['avg_fold_time_sec'].idxmax(), 'model']}) - {df['avg_fold_time_sec'].max():.1f}s por fold

RECOMENDACIONES:
{'-'*30}

1. Para máxima precisión: {df.loc[df['cv_acc_mean'].idxmax(), 'experiment']} ({df.loc[df['cv_acc_mean'].idxmax(), 'model']})
2. Para consistencia: {df.loc[df['cv_acc_std'].idxmin(), 'experiment']} ({df.loc[df['cv_acc_std'].idxmin(), 'model']})
3. Para eficiencia: {df.loc[df['avg_fold_time_sec'].idxmin(), 'experiment']} ({df.loc[df['avg_fold_time_sec'].idxmin(), 'model']})

NOTA: Todos los resultados fueron obtenidos mediante validación cruzada 
estratificada con conjuntos de test independientes, garantizando la 
validez estadística de las comparaciones.
"""

    with open(OUTPUT_DIR / "summary_report.txt", "w") as f:
        f.write(report)

    print(f"✓ Reporte resumen guardado en: {OUTPUT_DIR}/summary_report.txt")
    print("\nResumen ejecutivo:")
    print(report)


def main():
    print("🔍 Analizando resultados de validación cruzada...")

    # Cargar resultados
    results = load_all_results()
    if not results:
        print("❌ No se encontraron resultados de experimentos.")
        print("   Ejecuta primero: python experiments_cv.py")
        return

    print(f"📊 Encontrados {len(results)} experimentos")

    # Crear DataFrame resumen
    df = create_summary_dataframe(results)

    # Guardar CSV completo
    df.to_csv(OUTPUT_DIR / "cv_results_summary.csv", index=False)
    print(f"✓ CSV resumen guardado en: {OUTPUT_DIR}/cv_results_summary.csv")

    # Crear visualizaciones
    plot_cv_comparison(df)
    plot_cv_vs_test(df)
    plot_confidence_intervals(df)

    # Análisis estadístico con IC
    analyze_confidence_intervals(df)
    statistical_comparison(df, results)

    # Generar tabla LaTeX
    generate_latex_table(df)

    # Crear reporte resumen
    create_summary_report(df)

    print(f"\n🎉 Análisis completado. Resultados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
