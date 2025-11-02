import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats

warnings.filterwarnings("ignore")

# Configurar estilo de matplotlib
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Colores ANSI para resaltar
RESET = "\033[0m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RED = "\033[31m"

# Configuración de rutas
CONFIG_PATH = "experiments_yolo.yml"
EXPERIMENTS_DIR = (
    "/home/nahumfg/GithubProjects/parliament-voting-records/validation/yolo_columnas/experiments"
)
OUTPUT_DIR = (
    "/home/nahumfg/GithubProjects/parliament-voting-records/validation/yolo_columnas/evaluation"
)

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_experiment_config():
    """Cargar configuración de experimentos desde YAML"""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return [exp for exp in config["experiments"] if exp.get("execute", True)]


def get_best_metrics_from_csv(csv_path):
    """Extraer las mejores métricas de un archivo results.csv"""
    try:
        df = pd.read_csv(csv_path)

        # Filtrar filas con valores válidos (no inf, no nan)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(
            subset=[
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ]
        )

        if df.empty:
            return None

        # Obtener las mejores métricas
        best_metrics = {
            "precision": df["metrics/precision(B)"].max(),
            "recall": df["metrics/recall(B)"].max(),
            "mAP50": df["metrics/mAP50(B)"].max(),
            "mAP50_95": df["metrics/mAP50-95(B)"].max(),
            "best_epoch_precision": df.loc[df["metrics/precision(B)"].idxmax(), "epoch"],
            "best_epoch_recall": df.loc[df["metrics/recall(B)"].idxmax(), "epoch"],
            "best_epoch_mAP50": df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"],
            "best_epoch_mAP50_95": df.loc[df["metrics/mAP50-95(B)"].idxmax(), "epoch"],
        }

        return best_metrics

    except Exception as e:
        print(f"{RED}❌ Error procesando {csv_path}: {str(e)}{RESET}")
        return None


def calculate_confidence_interval(data, confidence=0.95):
    """Calcular intervalo de confianza para una muestra"""
    if len(data) < 2:
        return np.nan, np.nan

    mean = np.mean(data)
    std_err = stats.sem(data)  # Error estándar de la media
    h = std_err * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)

    return mean - h, mean + h


def collect_experiment_results():
    """Recopilar resultados de todos los experimentos"""
    experiments = load_experiment_config()
    folds = [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5",
        "fold_6",
        "fold_7",
        "fold_8",
        "fold_9",
        "fold_10",
    ]

    results = {}

    print(f"{BOLD}{BLUE}📊 Recopilando resultados de experimentos de votacion...{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n{BOLD}{CYAN}🔍 Procesando experimento: {exp_name}{RESET}")

        exp_results = {
            "precision": [],
            "recall": [],
            "mAP50": [],
            "mAP50_95": [],
            "fold_details": {},
        }

        # Recopilar resultados de todos los folds
        for fold in folds:
            fold_exp_name = f"{exp_name}_{fold}"
            results_path = os.path.join(EXPERIMENTS_DIR, fold_exp_name, "results.csv")

            if os.path.exists(results_path):
                metrics = get_best_metrics_from_csv(results_path)
                if metrics:
                    exp_results["precision"].append(metrics["precision"])
                    exp_results["recall"].append(metrics["recall"])
                    exp_results["mAP50"].append(metrics["mAP50"])
                    exp_results["mAP50_95"].append(metrics["mAP50_95"])
                    exp_results["fold_details"][fold] = metrics
                    print(
                        f"   ✅ {fold}: mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50_95']:.4f}"
                    )
                else:
                    print(f"   ❌ {fold}: Error procesando métricas")
            else:
                print(f"   ⚠️  {fold}: Archivo results.csv no encontrado")

        if exp_results["precision"]:  # Si hay al menos un resultado válido
            results[exp_name] = exp_results
            print(f"   {GREEN}📈 Total folds válidos: {len(exp_results['precision'])}/5{RESET}")
        else:
            print(f"   {RED}❌ No se encontraron resultados válidos para {exp_name}{RESET}")

    return results


def calculate_statistics(results):
    """Calcular estadísticas descriptivas e intervalos de confianza"""
    statistics = {}

    print(f"\n{BOLD}{BLUE}📊 Calculando estadísticas y intervalos de confianza...{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    for exp_name, exp_data in results.items():
        print(f"\n{BOLD}{YELLOW}📋 Experimento: {exp_name}{RESET}")

        exp_stats = {}

        for metric in ["precision", "recall", "mAP50", "mAP50_95"]:
            data = exp_data[metric]

            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data, ddof=1)  # Desviación estándar muestral
                min_val = np.min(data)
                max_val = np.max(data)

                # Intervalo de confianza del 95%
                ci_lower, ci_upper = calculate_confidence_interval(data, confidence=0.95)

                exp_stats[metric] = {
                    "values": data,
                    "count": len(data),
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "ci_95_lower": ci_lower,
                    "ci_95_upper": ci_upper,
                    "ci_95_width": ci_upper - ci_lower if not np.isnan(ci_upper) else np.nan,
                }

                print(
                    f"   {CYAN}{metric.upper():<12}: μ={mean_val:.4f} ± σ={std_val:.4f} | "
                    f"IC95%=[{ci_lower:.4f}, {ci_upper:.4f}] | "
                    f"Rango=[{min_val:.4f}, {max_val:.4f}]{RESET}"
                )
            else:
                exp_stats[metric] = None
                print(f"   {RED}{metric.upper():<12}: Sin datos válidos{RESET}")

        statistics[exp_name] = exp_stats

    return statistics


def save_detailed_results(results, statistics):
    """Guardar resultados detallados en archivos CSV y texto"""

    # 1. Guardar resumen estadístico
    summary_data = []
    for exp_name, exp_stats in statistics.items():
        for metric in ["precision", "recall", "mAP50", "mAP50_95"]:
            if exp_stats[metric] is not None:
                summary_data.append(
                    {
                        "experimento": exp_name,
                        "metrica": metric,
                        "media": exp_stats[metric]["mean"],
                        "desviacion_estandar": exp_stats[metric]["std"],
                        "minimo": exp_stats[metric]["min"],
                        "maximo": exp_stats[metric]["max"],
                        "ic_95_inferior": exp_stats[metric]["ci_95_lower"],
                        "ic_95_superior": exp_stats[metric]["ci_95_upper"],
                        "ancho_ic_95": exp_stats[metric]["ci_95_width"],
                        "num_folds": exp_stats[metric]["count"],
                    }
                )

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "resumen_estadistico_votacion.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"{GREEN}💾 Resumen estadístico guardado en: {summary_path}{RESET}")

    # 2. Guardar resultados por fold
    fold_data = []
    for exp_name, exp_data in results.items():
        for fold, fold_metrics in exp_data["fold_details"].items():
            fold_data.append(
                {
                    "experimento": exp_name,
                    "fold": fold,
                    "precision": fold_metrics["precision"],
                    "recall": fold_metrics["recall"],
                    "mAP50": fold_metrics["mAP50"],
                    "mAP50_95": fold_metrics["mAP50_95"],
                    "mejor_epoca_precision": fold_metrics["best_epoch_precision"],
                    "mejor_epoca_recall": fold_metrics["best_epoch_recall"],
                    "mejor_epoca_mAP50": fold_metrics["best_epoch_mAP50"],
                    "mejor_epoca_mAP50_95": fold_metrics["best_epoch_mAP50_95"],
                }
            )

    fold_df = pd.DataFrame(fold_data)
    fold_path = os.path.join(OUTPUT_DIR, "resultados_por_fold_votacion.csv")
    fold_df.to_csv(fold_path, index=False)
    print(f"{GREEN}💾 Resultados por fold guardados en: {fold_path}{RESET}")

    # 3. Guardar reporte detallado en texto
    report_path = os.path.join(OUTPUT_DIR, "reporte_detallado_votacion.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE DETALLADO - EVALUACIÓN DE EXPERIMENTOS DE VOTACION\n")
        f.write("=" * 70 + "\n\n")

        for exp_name, exp_stats in statistics.items():
            f.write(f"EXPERIMENTO: {exp_name}\n")
            f.write("-" * 50 + "\n")

            for metric in ["precision", "recall", "mAP50", "mAP50_95"]:
                if exp_stats[metric] is not None:
                    stats_data = exp_stats[metric]
                    f.write(f"\n{metric.upper()}:\n")
                    f.write(f"  • Número de folds: {stats_data['count']}\n")
                    f.write(f"  • Media: {stats_data['mean']:.6f}\n")
                    f.write(f"  • Desviación estándar: {stats_data['std']:.6f}\n")
                    f.write(f"  • Mínimo: {stats_data['min']:.6f}\n")
                    f.write(f"  • Máximo: {stats_data['max']:.6f}\n")
                    f.write(
                        f"  • Intervalo de confianza 95%: [{stats_data['ci_95_lower']:.6f}, {stats_data['ci_95_upper']:.6f}]\n"
                    )
                    f.write(f"  • Ancho del IC 95%: {stats_data['ci_95_width']:.6f}\n")
                    f.write(f"  • Valores por fold: {[f'{v:.6f}' for v in stats_data['values']]}\n")
                else:
                    f.write(f"\n{metric.upper()}: Sin datos válidos\n")

            f.write("\n" + "=" * 70 + "\n\n")

    print(f"{GREEN}💾 Reporte detallado guardado en: {report_path}{RESET}")


def generate_ranking():
    """Generar ranking de experimentos por mAP50-95"""
    summary_path = os.path.join(OUTPUT_DIR, "resumen_estadistico_votacion.csv")

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

        # Filtrar solo mAP50-95 y ordenar por media descendente
        map50_95_df = df[df["metrica"] == "mAP50_95"].copy()
        map50_95_df = map50_95_df.sort_values("media", ascending=False)

        print(f"\n{BOLD}{MAGENTA}🏆 RANKING DE EXPERIMENTOS POR mAP50-95{RESET}")
        print(f"{BOLD}{'='*80}{RESET}")

        for i, (_, row) in enumerate(map50_95_df.iterrows(), 1):
            exp_name = row["experimento"]
            mean_val = row["media"]
            std_val = row["desviacion_estandar"]
            ci_lower = row["ic_95_inferior"]
            ci_upper = row["ic_95_superior"]

            # Determinar color según posición
            if i == 1:
                color = f"{BOLD}{YELLOW}"  # Oro
            elif i == 2:
                color = f"{BOLD}{CYAN}"  # Plata
            elif i == 3:
                color = f"{BOLD}{MAGENTA}"  # Bronce
            else:
                color = f"{CYAN}"

            print(
                f"{color}#{i:2d}. {exp_name:<25} | "
                f"μ={mean_val:.4f} ± σ={std_val:.4f} | "
                f"IC95%=[{ci_lower:.4f}, {ci_upper:.4f}]{RESET}"
            )

        # Guardar ranking
        ranking_path = os.path.join(OUTPUT_DIR, "ranking_experimentos_votacion.csv")
        map50_95_df["posicion"] = range(1, len(map50_95_df) + 1)
        map50_95_df.to_csv(ranking_path, index=False)
        print(f"\n{GREEN}💾 Ranking guardado en: {ranking_path}{RESET}")


def create_metrics_comparison_plot(statistics):
    """Crear gráfica de comparación de métricas entre experimentos"""
    metrics_data = []

    for exp_name, exp_stats in statistics.items():
        for metric in ["precision", "recall", "mAP50", "mAP50_95"]:
            if exp_stats[metric] is not None:
                metrics_data.append(
                    {
                        "experimento": exp_name,
                        "metrica": metric,
                        "media": exp_stats[metric]["mean"],
                        "std": exp_stats[metric]["std"],
                        "ci_lower": exp_stats[metric]["ci_95_lower"],
                        "ci_upper": exp_stats[metric]["ci_95_upper"],
                    }
                )

    df_metrics = pd.DataFrame(metrics_data)

    # Crear subplots para cada métrica
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Comparación de Métricas por Experimento\n(Media ± Desviación Estándar con IC 95%)",
        fontsize=16,
        fontweight="bold",
    )

    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    metric_titles = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]

    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i // 2, i % 2]

        # Filtrar datos para esta métrica
        metric_data = df_metrics[df_metrics["metrica"] == metric].copy()
        metric_data = metric_data.sort_values("media", ascending=True)

        if not metric_data.empty:
            # Crear barplot con barras de error
            bars = ax.barh(
                range(len(metric_data)),
                metric_data["media"],
                xerr=metric_data["std"],
                capsize=5,
                alpha=0.7,
            )

            # Agregar intervalos de confianza como líneas
            for j, (_, row) in enumerate(metric_data.iterrows()):
                ax.plot([row["ci_lower"], row["ci_upper"]], [j, j], "r-", linewidth=2, alpha=0.8)
                ax.plot(
                    [row["ci_lower"], row["ci_lower"]],
                    [j - 0.1, j + 0.1],
                    "r-",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.plot(
                    [row["ci_upper"], row["ci_upper"]],
                    [j - 0.1, j + 0.1],
                    "r-",
                    linewidth=2,
                    alpha=0.8,
                )

            # Configurar ejes
            ax.set_yticks(range(len(metric_data)))
            ax.set_yticklabels(
                [exp.replace("_", "\n") for exp in metric_data["experimento"]], fontsize=8
            )
            ax.set_xlabel(f"{title} (Media ± σ)")
            ax.set_title(f"{title}")
            ax.grid(True, alpha=0.3)

            # Agregar valores en las barras
            for j, (bar, (_, row)) in enumerate(zip(bars, metric_data.iterrows())):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{row["media"]:.3f}',
                    ha="left",
                    va="center",
                    fontsize=8,
                )

    plt.tight_layout()

    # Guardar gráfica
    plot_path = os.path.join(OUTPUT_DIR, "comparacion_metricas_experimentos.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{GREEN}📊 Gráfica de comparación guardada en: {plot_path}{RESET}")


def create_fold_variability_plot(results):
    """Crear gráfica de variabilidad entre folds para cada experimento"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Variabilidad entre Folds por Experimento", fontsize=16, fontweight="bold")

    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    metric_titles = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]

    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i // 2, i % 2]

        # Preparar datos para boxplot
        data_for_boxplot = []
        labels = []

        for exp_name, exp_data in results.items():
            if exp_data[metric]:  # Si hay datos para esta métrica
                data_for_boxplot.append(exp_data[metric])
                labels.append(exp_name.replace("_", "\n"))

        if data_for_boxplot:
            # Crear boxplot
            bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)

            # Colorear las cajas
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Agregar puntos individuales
            for j, data in enumerate(data_for_boxplot):
                y = data
                x = np.random.normal(j + 1, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.6, s=30)

            ax.set_ylabel(f"{title}")
            ax.set_title(f"Distribución de {title} por Experimento")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Guardar gráfica
    plot_path = os.path.join(OUTPUT_DIR, "variabilidad_folds_experimentos.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{GREEN}📊 Gráfica de variabilidad guardada en: {plot_path}{RESET}")


def create_ranking_plot():
    """Crear gráfica del ranking de experimentos"""
    ranking_path = os.path.join(OUTPUT_DIR, "ranking_experimentos_votacion.csv")

    if os.path.exists(ranking_path):
        df_ranking = pd.read_csv(ranking_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Ranking de Experimentos por mAP@0.5:0.95", fontsize=16, fontweight="bold")

        # Gráfica 1: Barras horizontales con intervalos de confianza
        colors = [
            "gold" if i == 0 else "silver" if i == 1 else "orange" if i == 2 else "lightblue"
            for i in range(len(df_ranking))
        ]

        bars = ax1.barh(
            range(len(df_ranking)), df_ranking["media"], color=colors, alpha=0.8, edgecolor="black"
        )

        # Agregar intervalos de confianza
        ax1.errorbar(
            df_ranking["media"],
            range(len(df_ranking)),
            xerr=[
                df_ranking["media"] - df_ranking["ic_95_inferior"],
                df_ranking["ic_95_superior"] - df_ranking["media"],
            ],
            fmt="none",
            color="red",
            capsize=5,
            capthick=2,
        )

        # Configurar ejes
        ax1.set_yticks(range(len(df_ranking)))
        ax1.set_yticklabels(
            [f"#{i+1}. {exp.replace('_', ' ')}" for i, exp in enumerate(df_ranking["experimento"])],
            fontsize=10,
        )
        ax1.set_xlabel("mAP@0.5:0.95 (Media)")
        ax1.set_title("Ranking con Intervalos de Confianza 95%")
        ax1.grid(True, alpha=0.3)

        # Agregar valores en las barras
        for i, (bar, (_, row)) in enumerate(zip(bars, df_ranking.iterrows())):
            ax1.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{row["media"]:.4f}',
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Gráfica 2: Comparación de media vs desviación estándar
        scatter = ax2.scatter(
            df_ranking["media"],
            df_ranking["desviacion_estandar"],
            c=range(len(df_ranking)),
            cmap="RdYlBu_r",
            s=100,
            alpha=0.7,
            edgecolors="black",
        )

        # Agregar etiquetas a los puntos
        for _, row in df_ranking.iterrows():
            ax2.annotate(
                row["experimento"].replace("_", "\n"),
                (row["media"], row["desviacion_estandar"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                ha="left",
            )

        ax2.set_xlabel("Media mAP@0.5:0.95")
        ax2.set_ylabel("Desviación Estándar")
        ax2.set_title("Media vs Variabilidad")
        ax2.grid(True, alpha=0.3)

        # Agregar colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Posición en Ranking")

        plt.tight_layout()

        # Guardar gráfica
        plot_path = os.path.join(OUTPUT_DIR, "ranking_experimentos_grafica.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"{GREEN}📊 Gráfica de ranking guardada en: {plot_path}{RESET}")


def create_heatmap_metrics(statistics):
    """Crear heatmap de todas las métricas por experimento"""
    # Preparar datos para el heatmap
    heatmap_data = []

    for exp_name, exp_stats in statistics.items():
        row_data = {"experimento": exp_name}
        for metric in ["precision", "recall", "mAP50", "mAP50_95"]:
            if exp_stats[metric] is not None:
                row_data[f"{metric}_media"] = exp_stats[metric]["mean"]
                row_data[f"{metric}_std"] = exp_stats[metric]["std"]
            else:
                row_data[f"{metric}_media"] = np.nan
                row_data[f"{metric}_std"] = np.nan
        heatmap_data.append(row_data)

    df_heatmap = pd.DataFrame(heatmap_data)
    df_heatmap.set_index("experimento", inplace=True)

    # Crear subplots para media y desviación estándar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle("Heatmap de Métricas por Experimento", fontsize=16, fontweight="bold")

    # Heatmap de medias
    media_cols = [col for col in df_heatmap.columns if "_media" in col]
    df_media = df_heatmap[media_cols].copy()
    df_media.columns = [col.replace("_media", "").upper() for col in df_media.columns]

    sns.heatmap(
        df_media,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        ax=ax1,
        cbar_kws={"label": "Valor de la Métrica"},
    )
    ax1.set_title("Media de las Métricas")
    ax1.set_ylabel("Experimentos")

    # Heatmap de desviaciones estándar
    std_cols = [col for col in df_heatmap.columns if "_std" in col]
    df_std = df_heatmap[std_cols].copy()
    df_std.columns = [col.replace("_std", "").upper() for col in df_std.columns]

    sns.heatmap(
        df_std,
        annot=True,
        fmt=".4f",
        cmap="RdYlBu_r",
        ax=ax2,
        cbar_kws={"label": "Desviación Estándar"},
    )
    ax2.set_title("Desviación Estándar de las Métricas")
    ax2.set_ylabel("")

    plt.tight_layout()

    # Guardar gráfica
    plot_path = os.path.join(OUTPUT_DIR, "heatmap_metricas_experimentos.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{GREEN}📊 Heatmap de métricas guardado en: {plot_path}{RESET}")


def generate_all_plots(results, statistics):
    """Generar todas las gráficas"""
    print(f"\n{BOLD}{BLUE}📊 Generando gráficas de análisis...{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    # Crear todas las gráficas
    create_metrics_comparison_plot(statistics)
    create_fold_variability_plot(results)
    create_ranking_plot()
    create_heatmap_metrics(statistics)

    print(f"\n{GREEN}🎨 Todas las gráficas generadas exitosamente{RESET}")


def main():
    """Función principal"""
    print(f"{BOLD}{GREEN}🚀 EVALUACIÓN DE EXPERIMENTOS DE VOTACION{RESET}")
    print(f"{BOLD}{GREEN}Calculando desviaciones estándar e intervalos de confianza{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    # 1. Recopilar resultados
    results = collect_experiment_results()

    if not results:
        print(f"{RED}❌ No se encontraron resultados válidos para procesar{RESET}")
        return

    # 2. Calcular estadísticas
    statistics = calculate_statistics(results)

    # 3. Guardar resultados
    save_detailed_results(results, statistics)

    # 4. Generar ranking
    generate_ranking()

    # 5. Generar gráficas
    generate_all_plots(results, statistics)

    print(f"\n{BOLD}{GREEN}🎉 EVALUACIÓN COMPLETADA{RESET}")
    print(f"{GREEN}📁 Todos los archivos guardados en: {OUTPUT_DIR}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")


if __name__ == "__main__":
    main()
