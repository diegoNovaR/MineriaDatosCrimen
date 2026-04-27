"""
main.py
Punto de entrada del pipeline completo.
Ejecutar este archivo para lanzar carga + limpieza de los 3 datasets.

Uso (desde la raíz del proyecto o desde notebooks/):
    python main.py                        # procesa las 3 ciudades
    python main.py --city chicago         # procesa solo una ciudad
    python main.py --city chicago --info  # incluye info del raw
"""

import argparse
import sys
from pathlib import Path

# ─── Fix de rutas ─────────────────────────────────────────────────────────────
# Calcula la raíz del proyecto sin importar desde dónde se ejecute main.py
_THIS_FOLDER = Path(__file__).resolve().parent

# Si estamos dentro de notebooks/, subimos un nivel para llegar a la raíz
if _THIS_FOLDER.name == "notebooks":
    PROJECT_ROOT = _THIS_FOLDER.parent
else:
    PROJECT_ROOT = _THIS_FOLDER

sys.path.insert(0, str(PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

from config.settings import RAW_FILES
from src.loader import load_csv_chunked, get_basic_info
from src.cleaner import clean_city


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de crimen: carga y limpieza")
    parser.add_argument(
        "--city",
        choices=list(RAW_FILES.keys()),
        default=None,
        help="Procesar solo una ciudad (por defecto: todas)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Mostrar info básica del dataset crudo antes de limpiar",
    )
    return parser.parse_args()


def run_pipeline(city: str, show_info: bool = False):
    print(f"\n{'#'*55}")
    print(f"  PIPELINE: {city.upper()}")
    print(f"{'#'*55}")

    # 1. Carga
    df_raw = load_csv_chunked(city)

    # 2. Info opcional
    if show_info:
        get_basic_info(df_raw, city)

    # 3. Limpieza y guardado
    df_clean = clean_city(city, df_raw, save=True)

    print(f"\n✓ Pipeline completado para {city}")
    print(f"  Filas finales : {len(df_clean):,}")
    print(f"  Columnas      : {list(df_clean.columns)}")
    return df_clean


def main():
    args = parse_args()

    cities = [args.city] if args.city else list(RAW_FILES.keys())

    for city in cities:
        run_pipeline(city, show_info=args.info)

    print("\n" + "="*55)
    print("  PIPELINE COMPLETO ✓")
    print(f"  CSVs limpios en: {PROJECT_ROOT / 'data' / 'processed'}")
    print("="*55)


if __name__ == "__main__":
    main()