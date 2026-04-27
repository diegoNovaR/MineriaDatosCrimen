"""
loader.py
Carga los CSVs raw usando chunking para no saturar la RAM.
Solo lee las columnas necesarias definidas en settings.py.
"""

import pandas as pd
from pathlib import Path
import sys

# Permitir imports desde la raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_FILES, COLUMNS_TO_READ, CHUNK_SIZE


def load_csv_chunked(city: str) -> pd.DataFrame:
    """
    Carga el CSV raw de una ciudad en chunks y retorna un único DataFrame.

    Parámetros
    ----------
    city : str
        Clave de ciudad: 'chicago', 'philadelphia' o 'san_francisco'

    Retorna
    -------
    pd.DataFrame con solo las columnas necesarias
    """
    filepath = RAW_FILES.get(city)
    if filepath is None:
        raise ValueError(f"Ciudad '{city}' no encontrada en RAW_FILES (settings.py)")
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    cols = COLUMNS_TO_READ.get(city)
    chunks = []
    total_rows = 0

    print(f"\n{'='*50}")
    print(f"Cargando: {city.upper()}")
    print(f"Archivo : {filepath.name}")
    print(f"Columnas: {cols}")
    print(f"{'='*50}")

    try:
        reader = pd.read_csv(
            filepath,
            usecols=cols,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            on_bad_lines="skip",   # saltar filas corruptas sin romper la carga
        )

        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            total_rows += len(chunk)
            # Reporte de progreso cada 10 chunks
            if (i + 1) % 10 == 0:
                print(f"  → Chunks procesados: {i+1} | Filas acumuladas: {total_rows:,}")

    except Exception as e:
        raise RuntimeError(f"Error al leer {filepath.name}: {e}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n✓ Carga completa | Total filas: {len(df):,} | Columnas: {list(df.columns)}")
    return df


def load_all_cities() -> dict[str, pd.DataFrame]:
    """
    Carga los 3 datasets raw y retorna un diccionario {ciudad: DataFrame}.
    """
    results = {}
    for city in RAW_FILES:
        results[city] = load_csv_chunked(city)
    return results


def get_basic_info(df: pd.DataFrame, city: str) -> None:
    """
    Imprime información básica de un DataFrame recién cargado.
    """
    print(f"\n── Info básica: {city.upper()} ──")
    print(f"  Forma         : {df.shape}")
    print(f"  Tipos de datos:\n{df.dtypes}")
    print(f"\n  Nulos por columna:\n{df.isnull().sum()}")
    print(f"\n  Muestra (5 filas):\n{df.head()}")