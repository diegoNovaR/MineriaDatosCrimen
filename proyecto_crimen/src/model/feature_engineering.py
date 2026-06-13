import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RUTA_PROCESSED    = os.path.join("data", "processed")
ARCHIVO_UNIFICADO = os.path.join(RUTA_PROCESSED, "crime_unified.csv")
ARCHIVO_FEATURES  = os.path.join(RUTA_PROCESSED, "crime_features.csv")

# Categorías con <1% de registros — se excluyen del vector
CATS_RARAS = {"cat_trata_personas", "cat_amenaza_acoso", "cat_arson", "cat_homicidio"}


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
    Vector de características: ~25 columnas para PCA.
    - Escala : hora, mes (StandardScaler)
    - OHE    : categoria_delito (sin categorías raras), dia_semana, ciudad
    - Elimina: id, fecha, anio, latitud, longitud,
               es_fin_semana, es_peligroso, periodo_dia
    """
    print(f"\n{'='*50}")
    print(f"  GENERANDO VECTOR DE CARACTERÍSTICAS")
    print(f"{'='*50}")

    df = df_unificado.copy()

    # 1. Eliminar columnas que no van al vector
    COLS_DROP = ["id", "fecha", "anio", "latitud", "longitud",
                 "es_fin_semana", "es_peligroso", "periodo_dia"]
    cols_drop = [c for c in COLS_DROP if c in df.columns]
    df = df.drop(columns=cols_drop)
    print(f"  [drop] Eliminadas: {cols_drop}")

    # 2. Estandarizar hora, mes y columnas de clima
    cols_escalar = [c for c in ["hora", "mes", "temperatura", "precipitacion", "viento"]
                    if c in df.columns]
    scaler   = StandardScaler()
    escalado = scaler.fit_transform(df[cols_escalar])
    for i, col in enumerate(cols_escalar):
        df[f"{col}_scaled"] = escalado[:, i]
    df = df.drop(columns=cols_escalar)
    print(f"  [scaler] Estandarizadas : {cols_escalar}")
    print(f"  [scaler] Media          : { {c: round(scaler.mean_[i], 4) for i, c in enumerate(cols_escalar)} }")
    print(f"  [scaler] Std            : { {c: round(scaler.scale_[i], 4) for i, c in enumerate(cols_escalar)} }")

    # 3. OHE categoria_delito — excluir categorías raras (<1%)
    if "categoria_delito" in df.columns:
        dummies = pd.get_dummies(df["categoria_delito"], prefix="cat").astype(int)
        dummies = dummies[[c for c in dummies.columns if c not in CATS_RARAS]]
        df = pd.concat([df.drop(columns=["categoria_delito"]), dummies], axis=1)
        print(f"  [ohe] categoria_delito → {list(dummies.columns)}")

    # 4. OHE dia_semana y ciudad
    for col, prefijo in [("dia_semana", "dia"), ("ciudad", "ciudad")]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=prefijo).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            print(f"  [ohe] {col} → {list(dummies.columns)}")

    print(f"\n  Total columnas del vector : {df.shape[1]}")
    print(f"  Columnas finales          : {list(df.columns)}")

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