"""
settings.py
Configuración central del proyecto. Aquí se definen rutas, 
nombres de columnas y parámetros globales. Sin hardcoding en el resto del código.
"""

from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────────────────────────
# Siempre apunta a la raíz del proyecto (carpeta que contiene /config, /src, /data)
# sin importar desde dónde se ejecute el script
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_RAW_DIR   = BASE_DIR / "data" / "raw"
DATA_PROC_DIR  = BASE_DIR / "data" / "processed"

RAW_FILES = {
    "chicago":       DATA_RAW_DIR / "ChicagoData.csv",
    "philadelphia":  DATA_RAW_DIR / "PhilaData.csv",
    "san_francisco": DATA_RAW_DIR / "SanFrancisco.csv",
}

PROCESSED_FILES = {
    "chicago":       DATA_PROC_DIR / "chicago_clean.csv",
    "philadelphia":  DATA_PROC_DIR / "philadelphia_clean.csv",
    "san_francisco": DATA_PROC_DIR / "san_francisco_clean.csv",
}

# ─── Parámetros de carga ──────────────────────────────────────────────────────
CHUNK_SIZE = 100_000   # filas por chunk (ajustar según RAM disponible)

# ─── Esquema estándar de salida ───────────────────────────────────────────────
# Estas son las columnas que tendrá cada CSV procesado
STANDARD_COLUMNS = [
    "city",
    "date",
    "year",
    "month",
    "crime_type",
    "description",
    "latitude",
    "longitude",
    "arrest",
    "location_desc",
]

# ─── Mapeo de columnas por ciudad ─────────────────────────────────────────────
# raw_col: nombre original en el CSV
# Si el valor es None, el campo se deriva o se rellena con NaN
COLUMN_MAP = {
    "chicago": {
        "date":          "Date",
        "year":          "Year",
        "month":         None,          # se extrae de Date
        "crime_type":    "Primary Type",
        "description":   "Description",
        "latitude":      "Latitude",
        "longitude":     "Longitude",
        "arrest":        "Arrest",
        "location_desc": "Location Description",
    },
    "philadelphia": {
        "date":          "dispatch_date_time",
        "year":          None,           # se extrae de dispatch_date_time
        "month":         None,           # se extrae de dispatch_date_time
        "crime_type":    "text_general_code",
        "description":   "ucr_general",
        "latitude":      "lat",
        "longitude":     "lng",
        "arrest":        None,           # no existe en este dataset
        "location_desc": "location_block",
    },
    "san_francisco": {
        "date":          "Incident Datetime",
        "year":          "Incident Year",
        "month":         None,           # se extrae de Incident Datetime
        "crime_type":    "Incident Category",
        "description":   "Incident Description",
        "latitude":      "Latitude",
        "longitude":     "Longitude",
        "arrest":        "Resolution",
        "location_desc": "Intersection",
    },
}

# Columnas a leer de cada CSV raw (solo las necesarias → ahorro de memoria)
COLUMNS_TO_READ = {
    "chicago": [
        "Date", "Year", "Primary Type", "Description",
        "Latitude", "Longitude", "Arrest", "Location Description",
    ],
    "philadelphia": [
        "dispatch_date_time", "text_general_code", "ucr_general",
        "lat", "lng", "location_block",
    ],
    "san_francisco": [
        "Incident Datetime", "Incident Year", "Incident Category",
        "Incident Description", "Latitude", "Longitude",
        "Resolution", "Intersection",
    ],
}

# ─── Filtros opcionales ───────────────────────────────────────────────────────
# Rango de años a conservar (None = sin filtro)
YEAR_RANGE = (2025, 2026)