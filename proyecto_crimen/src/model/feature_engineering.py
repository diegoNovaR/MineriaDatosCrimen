import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RUTA_PROCESSED   = os.path.join("data", "processed")
ARCHIVO_UNIFICADO = os.path.join(RUTA_PROCESSED, "crime_unified.csv")
ARCHIVO_FEATURES  = os.path.join(RUTA_PROCESSED, "crime_features.csv")

COLS_ESCALAR = ["hora", "mes"]

COLS_ELIMINAR_FEATURES = ["id", "fecha", "ciudad"]


def unificar(datasets: dict) -> pd.DataFrame:
    """Concatena los 3 datasets transformados en uno solo y lo exporta."""
    frames = []
    for ciudad, df in datasets.items():
        if "ciudad" not in df.columns:
            df = df.copy()
            df["ciudad"] = ciudad
        frames.append(df)

    df_unificado = pd.concat(frames, ignore_index=True)

    os.makedirs(RUTA_PROCESSED, exist_ok=True)
    df_unificado.to_csv(ARCHIVO_UNIFICADO, index=False, encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"  DATASET UNIFICADO")
    print(f"{'='*50}")
    print(f"  Total filas    : {len(df_unificado):,}")
    print(f"  Columnas       : {list(df_unificado.columns)}")
    print(f"  Filas por ciudad:")
    for ciudad, n in df_unificado["ciudad"].value_counts().items():
        print(f"    {ciudad:<20} {n:,}")
    print(f"  Exportado → {ARCHIVO_UNIFICADO}")

    return df_unificado


def generar_features(df_unificado: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el vector de características numérico desde el dataset unificado.
    - Elimina id, fecha, ciudad
    - Estandariza hora, mes, latitud, longitud (StandardScaler)
    - anio se conserva como entero sin escalar
    - One-hot encoding: categoria_delito, periodo_dia, dia_semana
    - Convierte es_fin_semana y es_peligroso a int
    """
    print(f"\n{'='*50}")
    print(f"  GENERANDO VECTOR DE CARACTERÍSTICAS")
    print(f"{'='*50}")

    df = df_unificado.copy()

    # 1. Booleanos a int ANTES de cualquier otra operación
    for col in ["es_fin_semana", "es_peligroso"]:
        if col in df.columns:
            df[col] = df[col].astype(bool).astype(int)

    # 2. Eliminar columnas no útiles para modelos (ciudad va como OHE, no se elimina aquí)
    cols_drop = [c for c in ["id", "fecha"] if c in df.columns]
    df = df.drop(columns=cols_drop)
    print(f"  [drop] Eliminadas: {cols_drop}")

    # 3. Estandarizar numéricas (hora, mes, latitud, longitud)
    cols_escalar = [c for c in COLS_ESCALAR if c in df.columns]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[cols_escalar])
    for i, col in enumerate(cols_escalar):
        df[f"{col}_scaled"] = df_scaled[:, i]
    df = df.drop(columns=cols_escalar)
    print(f"  [scaler] Estandarizadas: {cols_escalar}")
    print(f"  [scaler] Media  : { {c: round(scaler.mean_[i], 4) for i, c in enumerate(cols_escalar)} }")
    print(f"  [scaler] Std    : { {c: round(scaler.scale_[i], 4) for i, c in enumerate(cols_escalar)} }")

    # 4. One-hot encoding
    COLS_OHE = {
        "categoria_delito": "cat",
        "periodo_dia":      "periodo",
        "dia_semana":       "dia",
        "ciudad":           "ciudad",
    }
    for col, prefijo in COLS_OHE.items():
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=prefijo).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            print(f"  [ohe] {col} → {list(dummies.columns)}")

    # 5. anio se conserva como int sin escalar
    if "anio" in df.columns:
        df["anio"] = df["anio"].astype("Int64")
        print(f"  [anio] Conservado como entero sin escalar (valor actual: {df['anio'].unique()})")

    print(f"\n  Total columnas del vector: {df.shape[1]}")
    print(f"  Columnas finales: {list(df.columns)}")

    df.to_csv(ARCHIVO_FEATURES, index=False, encoding="utf-8")
    print(f"\n  Exportado → {ARCHIVO_FEATURES}")

    return df


def cargar_unificado() -> pd.DataFrame:
    """Carga el CSV unificado desde data/processed/."""
    if not os.path.exists(ARCHIVO_UNIFICADO):
        print(f"\n[ERROR] No existe: {ARCHIVO_UNIFICADO}")
        return pd.DataFrame()

    df = pd.read_csv(ARCHIVO_UNIFICADO, low_memory=False, parse_dates=["fecha"])
    for col in ["hora", "mes", "anio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["es_fin_semana"] = df["es_fin_semana"].astype(bool)
    df["es_peligroso"]  = df["es_peligroso"].astype(bool)

    print(f"  [carga] crime_unified.csv → {len(df):,} filas, {df.shape[1]} columnas")
    return df


def cargar_features() -> pd.DataFrame:
    """Carga el vector de características desde data/processed/."""
    if not os.path.exists(ARCHIVO_FEATURES):
        print(f"\n[ERROR] No existe: {ARCHIVO_FEATURES}")
        return pd.DataFrame()

    df = pd.read_csv(ARCHIVO_FEATURES, low_memory=False)
    print(f"  [carga] crime_features.csv → {len(df):,} filas, {df.shape[1]} columnas")
    return df