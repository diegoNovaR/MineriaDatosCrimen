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
    Ejemplos: '41.802549018', '41,802549018', '-87,667246428'
    """
    return (
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .pipe(pd.to_numeric, errors="coerce")
    )


# ─── Helpers de limpieza por ciudad ──────────────────────────────────────────

def _clean_chicago(df: pd.DataFrame) -> pd.DataFrame:
    col = COLUMN_MAP["chicago"]
    date_parsed = pd.to_datetime(df[col["date"]], errors="coerce", dayfirst=False)

    out = pd.DataFrame({
        "city"         : "chicago",                              # fix: escalar en constructor
        "date"         : date_parsed,
        "year"         : df[col["year"]].astype("Int64"),
        "month"        : date_parsed.dt.month,
        "crime_type"   : df[col["crime_type"]].str.strip().str.upper(),
        "description"  : df[col["description"]].str.strip(),
        "latitude"     : _parse_coord(df[col["latitude"]]),
        "longitude"    : _parse_coord(df[col["longitude"]]),
        "arrest"       : df[col["arrest"]].astype(str).str.strip().str.upper(),
        "location_desc": df[col["location_desc"]].fillna("UNKNOWN").str.strip(),  # fix nulos
    })
    return out


def _clean_philadelphia(df: pd.DataFrame) -> pd.DataFrame:
    col = COLUMN_MAP["philadelphia"]
    date_parsed = pd.to_datetime(df[col["date"]], errors="coerce", utc=True)
    date_parsed = date_parsed.dt.tz_localize(None)   # quitar timezone

    out = pd.DataFrame({
        "city"         : "philadelphia",                         # fix: escalar en constructor
        "date"         : date_parsed,
        "year"         : date_parsed.dt.year.astype("Int64"),
        "month"        : date_parsed.dt.month,
        "crime_type"   : df[col["crime_type"]].str.strip().str.upper(),
        "description"  : df[col["description"]].astype(str).str.strip(),
        "latitude"     : _parse_coord(df[col["latitude"]]),
        "longitude"    : _parse_coord(df[col["longitude"]]),
        "arrest"       : "N/A",                                  # fix: no existe en este dataset
        "location_desc": df[col["location_desc"]].str.strip(),
    })

    # Fix outliers geograficos (12 detectados en EDA)
    lat_min, lat_max = 39.85, 40.15
    lon_min, lon_max = -75.3, -74.95
    mask_valid = (
        out["latitude"].between(lat_min, lat_max) &
        out["longitude"].between(lon_min, lon_max)
    )
    n_out = (~mask_valid).sum()
    if n_out > 0:
        print(f"  [philadelphia] Eliminando {n_out} outliers geográficos")
    out = out[mask_valid]

    return out


def _clean_san_francisco(df: pd.DataFrame) -> pd.DataFrame:
    col = COLUMN_MAP["san_francisco"]
    date_parsed = pd.to_datetime(df[col["date"]], errors="coerce")

    out = pd.DataFrame({
        "city"         : "san_francisco",                        # fix: escalar en constructor
        "date"         : date_parsed,
        "year"         : df[col["year"]].astype("Int64"),
        "month"        : date_parsed.dt.month,
        "crime_type"   : df[col["crime_type"]].str.strip().str.upper(),
        "description"  : df[col["description"]].str.strip(),
        "latitude"     : _parse_coord(df[col["latitude"]]),
        "longitude"    : _parse_coord(df[col["longitude"]]),
        # Resolution no es arrest exacto; True si contiene "ARREST"
        "arrest"       : df[col["arrest"]].str.upper().str.contains("ARREST", na=False).astype(str),
        "location_desc": df[col["location_desc"]].str.strip(),
    })

    # Fix: eliminar filas sin crime_type (413 detectados en EDA)
    n_sin_tipo = out["crime_type"].isna().sum()
    if n_sin_tipo > 0:
        print(f"  [san_francisco] Eliminando {n_sin_tipo} filas sin crime_type")
        out = out.dropna(subset=["crime_type"])

    return out


# ─── Limpieza general (aplica a todos) ───────────────────────────────────────

def _general_clean(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Pasos comunes post-normalización:
    - Eliminar filas sin coordenadas
    - Eliminar filas sin fecha
    - Filtrar por rango de años
    - Eliminar duplicados
    """
    before = len(df)

    df = df.dropna(subset=["latitude", "longitude"])
    after_coords = len(df)

    df = df.dropna(subset=["date"])
    after_date = len(df)

    after_year = len(df)
    if YEAR_RANGE is not None:
        y_min, y_max = YEAR_RANGE
        df = df[df["year"].between(y_min, y_max)]
        after_year = len(df)

    df = df.drop_duplicates()
    after = len(df)

    df = df.reset_index(drop=True)

    print(f"  [{city}] Filas cargadas            : {before:>10,}")
    print(f"  [{city}] Sin coords (NaN)          : {before - after_coords:>10,} → quedan {after_coords:,}")
    print(f"  [{city}] Sin fecha  (NaN)          : {after_coords - after_date:>10,} → quedan {after_date:,}")
    print(f"  [{city}] Fuera de año {YEAR_RANGE} : {after_date - after_year:>10,} → quedan {after_year:,}")
    print(f"  [{city}] Duplicados                : {after_year - after:>10,} → quedan {after:,}")
    print(f"  [{city}] ✓ FILAS FINALES           : {after:>10,}")

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
    """
    cleaner = CITY_CLEANERS.get(city)
    if cleaner is None:
        raise ValueError(f"No hay limpiador definido para '{city}'")

    print(f"\n{'='*50}")
    print(f"Limpiando: {city.upper()}")
    print(f"{'='*50}")

    df_clean = cleaner(df_raw)
    df_clean = _general_clean(df_clean, city)
    df_clean = df_clean[STANDARD_COLUMNS]

    if save:
        out_path = PROCESSED_FILES[city]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(out_path, index=False)
        print(f"  ✓ Guardado en: {out_path}")

    return df_clean


def clean_all_cities(raw_dataframes: dict) -> dict:
    cleaned = {}
    for city, df_raw in raw_dataframes.items():
        cleaned[city] = clean_city(city, df_raw, save=True)
    return cleaned


def load_processed(city: str) -> pd.DataFrame:
    """
    Carga el CSV ya procesado. Usar esto en visualizaciones.
    """
    path = PROCESSED_FILES.get(city)
    if path is None:
        raise ValueError(f"Ciudad '{city}' no definida en PROCESSED_FILES")
    if not path.exists():
        raise FileNotFoundError(
            f"CSV procesado no encontrado: {path}\n"
            "Ejecuta primero el pipeline de limpieza (main.py)"
        )
    df = pd.read_csv(path, low_memory=False, parse_dates=["date"])
    print(f"✓ Cargado procesado [{city}]: {df.shape[0]:,} filas")
    return df