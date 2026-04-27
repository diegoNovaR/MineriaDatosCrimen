"""
cleaner.py
Limpia y normaliza cada DataFrame al esquema estándar definido en settings.py.
Genera un CSV procesado por ciudad en data/processed/.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    COLUMN_MAP,
    STANDARD_COLUMNS,
    PROCESSED_FILES,
    YEAR_RANGE,
)


# ─── Utilidad: parsear coordenadas con coma decimal ──────────────────────────

def _parse_coord(series: pd.Series) -> pd.Series:
    """
    Convierte coordenadas a float manejando tanto punto como coma decimal.
    Ejemplos válidos: '41.802549018', '41,802549018', '-87,667246428'
    """
    return (
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .pipe(pd.to_numeric, errors="coerce")
    )


# ─── Helpers de limpieza por ciudad ──────────────────────────────────────────

def _clean_chicago(df: pd.DataFrame) -> pd.DataFrame:
    col = COLUMN_MAP["chicago"]

    out = pd.DataFrame()
    out["city"]          = "chicago"
    out["date"]          = pd.to_datetime(df[col["date"]], errors="coerce", dayfirst=False)
    out["year"]          = df[col["year"]].astype("Int64")
    out["month"]         = out["date"].dt.month
    out["crime_type"]    = df[col["crime_type"]].str.strip().str.upper()
    out["description"]   = df[col["description"]].str.strip()
    out["latitude"]      = _parse_coord(df[col["latitude"]])   # fix coma decimal
    out["longitude"]     = _parse_coord(df[col["longitude"]])  # fix coma decimal
    out["arrest"]        = df[col["arrest"]].astype(str).str.strip().str.upper()
    out["location_desc"] = df[col["location_desc"]].str.strip()

    return out


def _clean_philadelphia(df: pd.DataFrame) -> pd.DataFrame:
    col = COLUMN_MAP["philadelphia"]

    out = pd.DataFrame()
    out["city"]          = "philadelphia"
    out["date"]          = pd.to_datetime(df[col["date"]], errors="coerce", utc=True)
    out["date"]          = out["date"].dt.tz_localize(None)   # quitar timezone
    out["year"]          = out["date"].dt.year.astype("Int64")
    out["month"]         = out["date"].dt.month
    out["crime_type"]    = df[col["crime_type"]].str.strip().str.upper()
    out["description"]   = df[col["description"]].astype(str).str.strip()
    out["latitude"]      = _parse_coord(df[col["latitude"]])
    out["longitude"]     = _parse_coord(df[col["longitude"]])
    out["arrest"]        = pd.NA    # no existe en este dataset
    out["location_desc"] = df[col["location_desc"]].str.strip()

    return out


def _clean_san_francisco(df: pd.DataFrame) -> pd.DataFrame:
    col = COLUMN_MAP["san_francisco"]

    out = pd.DataFrame()
    out["city"]          = "san_francisco"
    out["date"]          = pd.to_datetime(df[col["date"]], errors="coerce")
    out["year"]          = df[col["year"]].astype("Int64")
    out["month"]         = out["date"].dt.month
    out["crime_type"]    = df[col["crime_type"]].str.strip().str.upper()
    out["description"]   = df[col["description"]].str.strip()
    out["latitude"]      = _parse_coord(df[col["latitude"]])   # fix coma decimal
    out["longitude"]     = _parse_coord(df[col["longitude"]])  # fix coma decimal
    # Resolution no es exactamente arrest; marcamos TRUE si dice ARREST
    out["arrest"]        = df[col["arrest"]].str.upper().str.contains("ARREST", na=False).astype(str)
    out["location_desc"] = df[col["location_desc"]].str.strip()

    return out


# ─── Limpieza general (aplica a todos) ───────────────────────────────────────

def _general_clean(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Pasos comunes post-normalización:
    - Eliminar filas sin coordenadas
    - Eliminar filas sin fecha
    - Filtrar por rango de años
    - Eliminar duplicados
    - Resetear índice
    """
    before = len(df)

    # Sin coordenadas → inútiles para mapas
    df = df.dropna(subset=["latitude", "longitude"])
    after_coords = len(df)

    # Sin fecha → inútiles para análisis temporal
    df = df.dropna(subset=["date"])
    after_date = len(df)

    # Filtro de año
    after_year = len(df)
    if YEAR_RANGE is not None:
        y_min, y_max = YEAR_RANGE
        df = df[df["year"].between(y_min, y_max)]
        after_year = len(df)

    # Duplicados exactos
    df = df.drop_duplicates()
    after = len(df)

    df = df.reset_index(drop=True)

    # Reporte detallado para diagnosticar dónde se pierden filas
    print(f"  [{city}] Filas cargadas       : {before:>10,}")
    print(f"  [{city}] Sin coords (NaN)     : {before - after_coords:>10,} → quedan {after_coords:,}")
    print(f"  [{city}] Sin fecha (NaN)      : {after_coords - after_date:>10,} → quedan {after_date:,}")
    print(f"  [{city}] Fuera de año {YEAR_RANGE}: {after_date - after_year:>10,} → quedan {after_year:,}")
    print(f"  [{city}] Duplicados           : {after_year - after:>10,} → quedan {after:,}")
    print(f"  [{city}] ✓ FILAS FINALES      : {after:>10,}")

    return df


# ─── Función principal ────────────────────────────────────────────────────────

CITY_CLEANERS = {
    "chicago":       _clean_chicago,
    "philadelphia":  _clean_philadelphia,
    "san_francisco": _clean_san_francisco,
}


def clean_city(city: str, df_raw: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Limpia y normaliza el DataFrame de una ciudad al esquema estándar.

    Parámetros
    ----------
    city    : clave de ciudad
    df_raw  : DataFrame crudo retornado por loader.py
    save    : si True, guarda el CSV procesado en data/processed/

    Retorna
    -------
    pd.DataFrame con columnas = STANDARD_COLUMNS
    """
    cleaner = CITY_CLEANERS.get(city)
    if cleaner is None:
        raise ValueError(f"No hay limpiador definido para '{city}'")

    print(f"\n{'='*50}")
    print(f"Limpiando: {city.upper()}")
    print(f"{'='*50}")

    df_clean = cleaner(df_raw)
    df_clean = _general_clean(df_clean, city)

    # Garantizar orden de columnas estándar
    df_clean = df_clean[STANDARD_COLUMNS]

    if save:
        out_path = PROCESSED_FILES[city]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(out_path, index=False)
        print(f"  ✓ Guardado en: {out_path}")

    return df_clean


def clean_all_cities(raw_dataframes: dict) -> dict:
    """
    Limpia los 3 datasets y retorna {ciudad: DataFrame_limpio}.
    """
    cleaned = {}
    for city, df_raw in raw_dataframes.items():
        cleaned[city] = clean_city(city, df_raw, save=True)
    return cleaned


def load_processed(city: str) -> pd.DataFrame:
    """
    Carga directamente el CSV ya procesado de una ciudad.
    Usar esto en visualizaciones para no repetir la limpieza.
    """
    path = PROCESSED_FILES.get(city)
    if path is None:
        raise ValueError(f"Ciudad '{city}' no definida en PROCESSED_FILES")
    if not path.exists():
        raise FileNotFoundError(
            f"CSV procesado no encontrado: {path}\n"
            "Ejecuta primero el pipeline de limpieza."
        )
    df = pd.read_csv(path, low_memory=False, parse_dates=["date"])
    print(f"✓ Cargado procesado [{city}]: {df.shape[0]:,} filas")
    return df