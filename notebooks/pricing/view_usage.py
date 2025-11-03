import csv
import json
import os
from collections import defaultdict
from pathlib import Path

USAGE_JSON_PATH = "/home/nahumfg/GithubProjects/parliament-voting-records/extract_ocr"


def extract_year_from_date(fecha_str):
    """Extrae el año de una fecha en formato DD/MM/YYYY"""
    try:
        if fecha_str and isinstance(fecha_str, str):
            parts = fecha_str.split("/")
            if len(parts) == 3:
                year = parts[2].strip()
                # Validar que el año tenga formato válido (4 dígitos)
                if year and year.isdigit() and len(year) == 4:
                    return year
    except (AttributeError, IndexError, ValueError):
        pass
    return None


def calculate_costs(base_path):
    """Calcula la suma total de cost_usd y por subcarpeta, y conteo por año"""
    total_cost = 0.0
    subfolder_costs = {}
    year_counts = defaultdict(int)  # Conteo por año
    ilegible_paths = []  # Lista de paths de archivos ilegibles

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

                    # Si es la carpeta de encabezados, extraer el año
                    if subfolder_name == "encabezados" and "output" in data:
                        # Verificar que output sea un diccionario y no una cadena
                        output = data["output"]
                        is_ilegible = False
                        razon = ""

                        if isinstance(output, dict):
                            # Buscar "fecha" de forma case-insensitive
                            fecha_key = None
                            for key in output.keys():
                                if key.lower() == "fecha":
                                    fecha_key = key
                                    break

                            if fecha_key:
                                year = extract_year_from_date(output[fecha_key])
                                if year:
                                    year_counts[year] += 1
                                else:
                                    # Si no se pudo extraer el año, contar como ilegible
                                    is_ilegible = True
                                    razon = "Fecha con formato inválido"
                                    year_counts["ilegible"] += 1
                            else:
                                # Si output es dict pero no tiene fecha, contar como ilegible
                                is_ilegible = True
                                razon = "Output sin clave 'fecha'"
                                year_counts["ilegible"] += 1
                        else:
                            # Si output no es dict (es string u otro tipo), contar como ilegible
                            is_ilegible = True
                            razon = f"Output no es diccionario (tipo: {type(output).__name__})"
                            year_counts["ilegible"] += 1

                        if is_ilegible:
                            ilegible_paths.append({"path": str(json_file), "razon": razon})

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Si hay un error, lo ignoramos y continuamos
                print(f"Error procesando {json_file}: {e}")
                continue

        subfolder_costs[subfolder_name] = {"cost": subfolder_cost, "file_count": len(json_files)}

    return total_cost, subfolder_costs, dict(year_counts), ilegible_paths


def save_ilegible_csv(ilegible_paths, output_path):
    """Guarda los paths de archivos ilegibles en un CSV"""
    if not ilegible_paths:
        print("\nNo hay archivos ilegibles para guardar.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "razon"])
        writer.writeheader()
        writer.writerows(ilegible_paths)

    print(f"\n✓ CSV de archivos ilegibles guardado en: {output_path}")
    print(f"  Total de archivos ilegibles: {len(ilegible_paths)}")


def print_report(total_cost, subfolder_costs, year_counts):
    """Imprime el reporte de costos y conteo por año"""
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

    # Mostrar conteo por año si hay datos
    if year_counts:
        print("\n" + "=" * 80)
        print("CONTEO POR AÑO - ENCABEZADOS")
        print("=" * 80)
        print(f"\n{'Año':<15} {'Cantidad':<15}")
        print("-" * 80)

        # Ordenar por año descendente, poniendo "ilegible" al final
        years_numericos = [
            (year, count) for year, count in year_counts.items() if year != "ilegible"
        ]
        years_numericos.sort(key=lambda x: x[0], reverse=True)  # Ordenar años descendente

        # Agregar "ilegible" al final si existe
        if "ilegible" in year_counts:
            years_numericos.append(("ilegible", year_counts["ilegible"]))

        sorted_years = years_numericos

        for year, count in sorted_years:
            print(f"{year:<15} {count:<15}")

        print("-" * 80)
        print(f"{'TOTAL':<15} {sum(year_counts.values()):<15}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    total_cost, subfolder_costs, year_counts, ilegible_paths = calculate_costs(USAGE_JSON_PATH)
    print_report(total_cost, subfolder_costs, year_counts)

    # Guardar CSV de archivos ilegibles
    script_dir = Path(__file__).parent
    csv_output_path = script_dir / "archivos_ilegibles.csv"
    save_ilegible_csv(ilegible_paths, csv_output_path)
