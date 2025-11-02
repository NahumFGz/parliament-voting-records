import json
import os
from pathlib import Path

USAGE_JSON_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/extract_ocr"


def calculate_costs(base_path):
    """Calcula la suma total de cost_usd y por subcarpeta"""
    total_cost = 0.0
    subfolder_costs = {}

    # Iterar sobre todas las subcarpetas
    base_path = Path(base_path)
    for subfolder in base_path.iterdir():
        if not subfolder.is_dir():
            continue

        subfolder_name = subfolder.name
        subfolder_cost = 0.0

        # Buscar todos los archivos JSON en la subcarpeta
        json_files = list(subfolder.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Verificar si existe la clave 'meta' y 'cost_usd'
                    if "meta" in data and "cost_usd" in data["meta"]:
                        cost = float(data["meta"]["cost_usd"])
                        subfolder_cost += cost
                        total_cost += cost

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Si hay un error, lo ignoramos y continuamos
                print(f"Error procesando {json_file}: {e}")
                continue

        subfolder_costs[subfolder_name] = {"cost": subfolder_cost, "file_count": len(json_files)}

    return total_cost, subfolder_costs


def print_report(total_cost, subfolder_costs):
    """Imprime el reporte de costos"""
    print("\n" + "=" * 80)
    print("REPORTE DE USO - COSTOS USD")
    print("=" * 80)
    print(f"\n{'Subcarpeta':<30} {'Archivos':<15} {'Costo (USD)':<15}")
    print("-" * 80)

    # Ordenar por costo descendente
    sorted_subfolders = sorted(subfolder_costs.items(), key=lambda x: x[1]["cost"], reverse=True)

    for subfolder_name, data in sorted_subfolders:
        print(f"{subfolder_name:<30} {data['file_count']:<15} ${data['cost']:.4f}")

    print("-" * 80)
    print(
        f"{'TOTAL':<30} {sum(d['file_count'] for d in subfolder_costs.values()):<15} ${total_cost:.4f}"
    )
    print("=" * 80)
    print(f"\nTotal costo: ${total_cost:.4f} USD")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    total_cost, subfolder_costs = calculate_costs(USAGE_JSON_PATH)
    print_report(total_cost, subfolder_costs)
