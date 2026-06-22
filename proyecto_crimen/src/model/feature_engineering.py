import os
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler

RUTA_PROCESSED    = os.path.join("data", "processed")
ARCHIVO_UNIFICADO = os.path.join(RUTA_PROCESSED, "crime_unified.csv")
ARCHIVO_FEATURES  = os.path.join(RUTA_PROCESSED, "crime_features.csv")

# Categorías con <1% de registros — se excluyen del vector
CATS_RARAS = {"cat_trata_personas", "cat_amenaza_acoso", "cat_arson", "cat_homicidio"}

# Mapeo ciudad -> estado para feriados específicos
ESTADO_POR_CIUDAD = {
    "chicago":       "IL",
    "philadelphia":  "PA",
    "san_francisco": "CA",
}


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


def _agregar_es_feriado(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna es_feriado (0/1) según el estado de cada ciudad."""
    anios = df["fecha"].dt.year.unique().tolist()
    calendarios = {
        ciudad: holidays.US(state=estado, years=anios)
        for ciudad, estado in ESTADO_POR_CIUDAD.items()
    }

    def check_feriado(row):
        cal = calendarios.get(row["ciudad"])
        if cal is None:
            return 0
        return int(row["fecha"].date() in cal)

    df["es_feriado"] = df.apply(check_feriado, axis=1)
    n_feriados = df["es_feriado"].sum()
    print(f"  [feriado] Registros en día feriado: {n_feriados:,} "
          f"({n_feriados/len(df)*100:.2f}%)")
    return df


def generar_features(df_unificado: pd.DataFrame) -> pd.DataFrame:
    """
    Vector de características para PCA/UMAP.
    - Escala : hora, mes, temperatura, viento (StandardScaler)
    - OHE    : categoria_delito (sin categorías raras), dia_semana, ciudad
    - Agrega : es_feriado (0/1) calculado con librería holidays
    - Elimina: id, fecha, anio, latitud, longitud,
               es_fin_semana, periodo_dia, precipitacion
    """
    print(f"\n{'='*50}")
    print(f"  GENERANDO VECTOR DE CARACTERÍSTICAS")
    print(f"{'='*50}")

    df = df_unificado.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Filtrar sin_relevancia ANTES de generar el vector
    antes = len(df)
    df = df[df["categoria_delito"] != "sin_relevancia"].copy()
    print(f"  [filtro] sin_relevancia eliminados: {antes - len(df):,} "
          f"({(antes - len(df))/antes*100:.1f}%)")
    print(f"  [filtro] Registros restantes: {len(df):,}")

    # 1. Agregar es_feriado ANTES de eliminar fecha
    df = _agregar_es_feriado(df)

    # 2. Eliminar columnas que no van al vector
    COLS_DROP = ["id", "fecha", "anio", "latitud", "longitud",
                 "es_fin_semana", "periodo_dia", "precipitacion"]
    cols_drop = [c for c in COLS_DROP if c in df.columns]
    df = df.drop(columns=cols_drop)
    print(f"  [drop] Eliminadas: {cols_drop}")

    # 3. Estandarizar hora, mes, temperatura, viento
    cols_escalar = [c for c in ["hora", "mes", "temperatura", "viento"]
                    if c in df.columns]
    scaler   = StandardScaler()
    escalado = scaler.fit_transform(df[cols_escalar])
    for i, col in enumerate(cols_escalar):
        df[f"{col}_scaled"] = escalado[:, i]
    df = df.drop(columns=cols_escalar)
    print(f"  [scaler] Estandarizadas : {cols_escalar}")
    print(f"  [scaler] Media          : { {c: round(scaler.mean_[i], 4) for i, c in enumerate(cols_escalar)} }")
    print(f"  [scaler] Std            : { {c: round(scaler.scale_[i], 4) for i, c in enumerate(cols_escalar)} }")

    # 4. OHE categoria_delito — excluir categorías raras (<1%)
    if "categoria_delito" in df.columns:
        dummies = pd.get_dummies(df["categoria_delito"], prefix="cat").astype(int)
        dummies = dummies[[c for c in dummies.columns if c not in CATS_RARAS]]
        df = pd.concat([df.drop(columns=["categoria_delito"]), dummies], axis=1)
        print(f"  [ohe] categoria_delito → {list(dummies.columns)}")

    # 5. OHE dia_semana y ciudad
    for col, prefijo in [("dia_semana", "dia"), ("ciudad", "ciudad")]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=prefijo).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            print(f"  [ohe] {col} → {list(dummies.columns)}")

    # 6. es_peligroso a int (booleano original)
    if "es_peligroso" in df.columns:
        df["es_peligroso"] = df["es_peligroso"].astype(bool).astype(int)

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