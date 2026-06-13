import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap

RUTA_PROCESSED   = os.path.join("data", "processed")
RUTA_OUTPUTS     = os.path.join("outputs", "umap")
ARCHIVO_UMAP_2D  = os.path.join(RUTA_PROCESSED, "crime_umap_2d.csv")

N_NEIGHBORS    = 15
MIN_DIST       = 0.1
N_COMPONENTS   = 2
RANDOM_STATE   = 42
METRIC         = "euclidean"

# Filas para ENTRENAR el modelo (estratificado por ciudad)
N_ENTRENAMIENTO = 50_000

COLS_CONTINUAS = ["hora_scaled", "mes_scaled", "es_peligroso",
                  "temperatura_scaled", "precipitacion_scaled", "viento_scaled"]


def _construir_matriz(df_features: pd.DataFrame,
                      df_unificado: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de entrada combinando features + lat/lon."""
    df_entrada = pd.DataFrame()

    for col in ["latitud", "longitud"]:
        if col in df_unificado.columns:
            df_entrada[col] = df_unificado[col].values

    for col in COLS_CONTINUAS:
        if col in df_features.columns:
            df_entrada[col] = df_features[col].values

    cols_ohe = [c for c in df_features.columns
                if c.startswith(("cat_", "dia_", "ciudad_"))]
    for col in cols_ohe:
        df_entrada[col] = df_features[col].values

    # Eliminar varianza 0
    cols_var_cero = df_entrada.columns[df_entrada.std() == 0].tolist()
    if cols_var_cero:
        df_entrada = df_entrada.drop(columns=cols_var_cero)

    # Rellenar nulos con media
    if df_entrada.isnull().sum().sum() > 0:
        df_entrada = df_entrada.fillna(df_entrada.mean())

    return df_entrada


def _muestra_estratificada(df_entrada: pd.DataFrame,
                            df_unificado: pd.DataFrame,
                            n: int) -> np.ndarray:
    """Índices de muestra estratificada por ciudad."""
    ciudades = df_unificado["ciudad"].unique()
    n_por_ciudad = n // len(ciudades)
    indices = []
    rng = np.random.default_rng(RANDOM_STATE)

    for ciudad in ciudades:
        idx_ciudad = df_unificado.index[df_unificado["ciudad"] == ciudad].tolist()
        n_tomar    = min(n_por_ciudad, len(idx_ciudad))
        elegidos   = rng.choice(idx_ciudad, size=n_tomar, replace=False)
        indices.extend(elegidos)
        print(f"    {ciudad:<20} {n_tomar:,} filas")

    return np.array(indices)


def aplicar_umap(df_features: pd.DataFrame,
                 df_unificado: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica UMAP 2D con muestra estratificada para entrenar
    y transform() para proyectar toda la data.
    """
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    os.makedirs(RUTA_PROCESSED, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  UMAP — REDUCCIÓN DE DIMENSIONALIDAD A 2D")
    print(f"{'='*55}")

    n = min(len(df_features), len(df_unificado))
    df_feat = df_features.iloc[:n].reset_index(drop=True)
    df_unif = df_unificado.iloc[:n].reset_index(drop=True)

    # Construir matriz completa
    X = _construir_matriz(df_feat, df_unif)

    print(f"  Columnas entrada : {X.shape[1]}")
    print(f"  Filas totales    : {len(X):,}")
    print(f"  Filas entrenamiento: {N_ENTRENAMIENTO:,} (estratificado por ciudad)")
    print(f"  Parámetros       : n_neighbors={N_NEIGHBORS}, "
          f"min_dist={MIN_DIST}, metric={METRIC}")

    # Muestra estratificada para entrenamiento
    print(f"\n  [Muestra de entrenamiento por ciudad]")
    idx_train = _muestra_estratificada(X, df_unif, N_ENTRENAMIENTO)
    X_train   = X.iloc[idx_train].values

    print(f"\n  Entrenando UMAP con {len(X_train):,} filas...")
    reducer = umap.UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=METRIC,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    reducer.fit(X_train)

    # Proyectar toda la data
    print(f"\n  Proyectando {len(X):,} filas con transform()...")
    X_umap = reducer.transform(X.values)

    df_umap = pd.DataFrame(X_umap, columns=["umap1", "umap2"])
    df_umap.to_csv(ARCHIVO_UMAP_2D, index=False, encoding="utf-8")

    # Verificar
    verificacion = pd.read_csv(ARCHIVO_UMAP_2D)
    print(f"\n  Filas exportadas : {len(verificacion):,}")
    print(f"  Nulos en CSV     : {verificacion.isnull().sum().sum()}")
    print(f"  Exportado → {ARCHIVO_UMAP_2D}")

    _grafica_rapida(df_umap)
    return df_umap


def cargar_umap_2d() -> pd.DataFrame:
    if not os.path.exists(ARCHIVO_UMAP_2D):
        print(f"\n[ERROR] No existe: {ARCHIVO_UMAP_2D}")
        return pd.DataFrame()
    df = pd.read_csv(ARCHIVO_UMAP_2D, low_memory=False)
    print(f"  [carga] crime_umap_2d.csv → {len(df):,} filas, {df.shape[1]} columnas")
    return df


def _grafica_rapida(df_umap: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df_umap["umap1"], df_umap["umap2"],
               s=0.5, alpha=0.3, color="#555555")
    ax.set_title(
        f"UMAP 2D — Vista general\n"
        f"n_neighbors={N_NEIGHBORS}  |  min_dist={MIN_DIST}  |  "
        f"{len(df_umap):,} puntos",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    ruta = os.path.join(RUTA_OUTPUTS, "umap_2d_general.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")