"""
analyzer.py
EDA (Análisis Exploratorio de Datos) por ciudad.
Lee los CSVs procesados de data/processed/ y muestra resultados en consola.

Uso directo:
    python src/analyzer.py                  # analiza las 3 ciudades
    python src/analyzer.py --city chicago   # analiza solo una ciudad
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_FILES, YEAR_RANGE

# Límites geográficos válidos por ciudad (bbox aproximado)
# Sirve para detectar coordenadas outlier / erróneas
GEO_BOUNDS = {
    "chicago": {
        "lat": (41.6, 42.1),
        "lon": (-87.95, -87.5),
    },
    "philadelphia": {
        "lat": (39.85, 40.15),
        "lon": (-75.3, -74.95),
    },
    "san_francisco": {
        "lat": (37.6, 37.95),
        "lon": (-122.55, -122.33),
    },
}

SEPARATOR     = "=" * 60
SEPARATOR_MID = "-" * 60


# ─── Helpers de presentación ──────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _section(title: str) -> None:
    print(f"\n{SEPARATOR_MID}")
    print(f"  {title}")
    print(SEPARATOR_MID)


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{value / total * 100:.2f}%"


# ─── Bloques de análisis ──────────────────────────────────────────────────────

def resumen_general(df: pd.DataFrame, city: str) -> None:
    _section("1. RESUMEN GENERAL")

    total = len(df)
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    print(f"  Ciudad           : {city.upper()}")
    print(f"  Total filas      : {total:,}")
    print(f"  Columnas         : {list(df.columns)}")
    print(f"  Memoria usada    : {mem_mb:.2f} MB")

    if "date" in df.columns and df["date"].notna().any():
        print(f"  Fecha mínima     : {df['date'].min()}")
        print(f"  Fecha máxima     : {df['date'].max()}")

    if "year" in df.columns:
        years = sorted(df["year"].dropna().unique().astype(int).tolist())
        print(f"  Años presentes   : {years}")

    print(f"\n  Tipos de datos:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<20} {dtype}")


def calidad_datos(df: pd.DataFrame, city: str) -> None:
    _section("2. CALIDAD DE DATOS — NULOS")

    total = len(df)
    print(f"  {'Columna':<20} {'Nulos':>8} {'Porcentaje':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*12}")

    for col in df.columns:
        n_null = df[col].isna().sum()
        print(f"  {col:<20} {n_null:>8,} {_pct(n_null, total):>12}")


def outliers_geograficos(df: pd.DataFrame, city: str) -> None:
    _section("3. OUTLIERS GEOGRÁFICOS")

    bounds = GEO_BOUNDS.get(city)
    if bounds is None:
        print(f"  Sin límites definidos para {city}")
        return

    lat_min, lat_max = bounds["lat"]
    lon_min, lon_max = bounds["lon"]

    mask_lat = (df["latitude"] < lat_min) | (df["latitude"] > lat_max)
    mask_lon = (df["longitude"] < lon_min) | (df["longitude"] > lon_max)
    mask_out = mask_lat | mask_lon

    total      = len(df)
    n_out      = mask_out.sum()
    n_lat_out  = mask_lat.sum()
    n_lon_out  = mask_lon.sum()

    print(f"  Rango latitud  válido : [{lat_min}, {lat_max}]")
    print(f"  Rango longitud válido : [{lon_min}, {lon_max}]")
    print(f"  Latitud  fuera rango  : {n_lat_out:,}  ({_pct(n_lat_out, total)})")
    print(f"  Longitud fuera rango  : {n_lon_out:,}  ({_pct(n_lon_out, total)})")
    print(f"  Total outliers geo    : {n_out:,}  ({_pct(n_out, total)})")

    if n_out > 0:
        print(f"\n  Estadísticas de coordenadas (incluyendo outliers):")
        for col in ["latitude", "longitude"]:
            s = df[col]
            print(f"    {col}: min={s.min():.5f}  max={s.max():.5f}  "
                  f"media={s.mean():.5f}  std={s.std():.5f}")


def distribucion_temporal(df: pd.DataFrame, city: str) -> None:
    _section("4. DISTRIBUCIÓN TEMPORAL")

    # Por año
    if "year" in df.columns:
        print(f"\n  Crímenes por año:")
        por_anio = (
            df.groupby("year", observed=True)
              .size()
              .reset_index(name="total")
              .sort_values("year")
        )
        total = por_anio["total"].sum()
        print(f"  {'Año':<8} {'Total':>10} {'%':>8}")
        print(f"  {'-'*8} {'-'*10} {'-'*8}")
        for _, row in por_anio.iterrows():
            print(f"  {int(row['year']):<8} {int(row['total']):>10,} {_pct(int(row['total']), total):>8}")

    # Por mes (global, todos los años)
    if "month" in df.columns:
        meses = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        print(f"\n  Crímenes por mes (todos los años):")
        por_mes = (
            df.groupby("month", observed=True)
              .size()
              .reset_index(name="total")
              .sort_values("month")
        )
        total = por_mes["total"].sum()
        print(f"  {'Mes':<14} {'Total':>10} {'%':>8}")
        print(f"  {'-'*14} {'-'*10} {'-'*8}")
        for _, row in por_mes.iterrows():
            nombre = meses.get(int(row["month"]), str(int(row["month"])))
            print(f"  {nombre:<14} {int(row['total']):>10,} {_pct(int(row['total']), total):>8}")


def distribucion_crimenes(df: pd.DataFrame, city: str, top_n: int = 10) -> None:
    _section(f"5. TOP {top_n} TIPOS DE CRIMEN")

    if "crime_type" not in df.columns:
        print("  Columna crime_type no encontrada.")
        return

    total = len(df)
    top = (
        df["crime_type"]
          .value_counts()
          .head(top_n)
          .reset_index()
    )
    top.columns = ["crime_type", "total"]

    print(f"  {'#':<4} {'Tipo de crimen':<35} {'Total':>10} {'%':>8}")
    print(f"  {'-'*4} {'-'*35} {'-'*10} {'-'*8}")
    for i, row in top.iterrows():
        print(f"  {i+1:<4} {str(row['crime_type']):<35} {int(row['total']):>10,} {_pct(int(row['total']), total):>8}")


def resumen_arrest(df: pd.DataFrame, city: str) -> None:
    _section("6. ARRESTOS")

    if "arrest" not in df.columns or df["arrest"].isna().all():
        print(f"  Sin datos de arresto para {city}.")
        return

    conteo = df["arrest"].value_counts(dropna=False)
    total  = len(df)

    print(f"  {'Valor':<20} {'Total':>10} {'%':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*8}")
    for val, cnt in conteo.items():
        print(f"  {str(val):<20} {cnt:>10,} {_pct(cnt, total):>8}")


# ─── Función principal ────────────────────────────────────────────────────────

def analyze_city(city: str) -> None:
    """
    Ejecuta el EDA completo de una ciudad y lo imprime en consola.
    """
    path = PROCESSED_FILES.get(city)
    if path is None:
        print(f"[ERROR] Ciudad '{city}' no definida en PROCESSED_FILES")
        return
    if not path.exists():
        print(f"[ERROR] CSV procesado no encontrado: {path}")
        print("        Ejecuta primero el pipeline de limpieza (main.py)")
        return

    df = pd.read_csv(path, low_memory=False, parse_dates=["date"])

    _header(f"EDA — {city.upper()}")
    resumen_general(df, city)
    calidad_datos(df, city)
    outliers_geograficos(df, city)
    distribucion_temporal(df, city)
    distribucion_crimenes(df, city)
    resumen_arrest(df, city)

    print(f"\n{SEPARATOR}")
    print(f"  FIN EDA — {city.upper()}")
    print(SEPARATOR)


def analyze_all() -> None:
    for city in PROCESSED_FILES:
        analyze_city(city)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EDA del proyecto de crimen")
    parser.add_argument(
        "--city",
        choices=list(PROCESSED_FILES.keys()),
        default=None,
        help="Ciudad a analizar (por defecto: todas)",
    )
    args = parser.parse_args()

    if args.city:
        analyze_city(args.city)
    else:
        analyze_all()