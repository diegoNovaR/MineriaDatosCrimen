import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RUTA_PROCESSED   = os.path.join("data", "processed")
RUTA_OUTPUTS     = os.path.join("outputs", "tsne")
ARCHIVO_TSNE_2D  = os.path.join(RUTA_PROCESSED, "crime_tsne_2d.csv")
ARCHIVO_IDX_TSNE = os.path.join(RUTA_PROCESSED, "crime_tsne_indices.csv")

# PCA
PCA_VARIANZA = 0.90

# t-SNE
PERPLEXITY   = 30
N_ITER       = 1000
RANDOM_STATE = 42
N_COMPONENTS = 2

# Muestra estratificada (t-SNE no tiene transform())
N_MUESTRA = 30_000

COLS_CONTINUAS = ["hora_scaled", "mes_scaled", "es_peligroso", "es_feriado",
                  "temperatura_scaled", "viento_scaled"]


def _construir_matriz(df_features: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de entrada desde el vector de características."""
    df_entrada = pd.DataFrame()

    for col in COLS_CONTINUAS:
        if col in df_features.columns:
            df_entrada[col] = df_features[col].values

    cols_ohe = [c for c in df_features.columns
                if c.startswith(("cat_", "dia_")) and not c.startswith("ciudad_")]
    for col in cols_ohe:
        df_entrada[col] = df_features[col].values

    cols_var_cero = df_entrada.columns[df_entrada.std() == 0].tolist()
    if cols_var_cero:
        df_entrada = df_entrada.drop(columns=cols_var_cero)

    if df_entrada.isnull().sum().sum() > 0:
        df_entrada = df_entrada.fillna(df_entrada.mean())

    return df_entrada


def _muestra_estratificada(df_features: pd.DataFrame,
                            df_unificado: pd.DataFrame) -> tuple:
    """Muestra estratificada por ciudad. Retorna (X_muestra, indices)."""
    ciudades     = df_unificado["ciudad"].unique()
    n_por_ciudad = N_MUESTRA // len(ciudades)
    indices = []
    rng = np.random.default_rng(RANDOM_STATE)

    for ciudad in ciudades:
        idx_ciudad = df_unificado.index[df_unificado["ciudad"] == ciudad].tolist()
        n_tomar    = min(n_por_ciudad, len(idx_ciudad))
        elegidos   = rng.choice(idx_ciudad, size=n_tomar, replace=False)
        indices.extend(elegidos.tolist())
        print(f"    {ciudad:<20} {n_tomar:,} filas")

    indices = np.array(indices)
    return df_features.iloc[indices], indices


def aplicar_tsne(df_features: pd.DataFrame,
                 df_unificado: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline PCA + t-SNE:
    1. PCA reduce el vector a componentes con PCA_VARIANZA de varianza
    2. Muestra estratificada de N_MUESTRA filas
    3. t-SNE proyecta la muestra a 2D
    Exporta crime_tsne_2d.csv (coordenadas) y crime_tsne_indices.csv (índices originales)
    """
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    os.makedirs(RUTA_PROCESSED, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  PCA + t-SNE — REDUCCIÓN DE DIMENSIONALIDAD A 2D")
    print(f"{'='*55}")

    n = min(len(df_features), len(df_unificado))
    df_feat = df_features.iloc[:n].reset_index(drop=True)
    df_unif = df_unificado.iloc[:n].reset_index(drop=True)

    # Construir matriz
    X = _construir_matriz(df_feat)
    print(f"  Columnas entrada (pre-PCA) : {X.shape[1]}")
    print(f"  Filas totales              : {len(X):,}")

    # ── Paso 1: PCA ──────────────────────────────────────────────────────
    print(f"\n  [PASO 1] PCA — varianza objetivo={PCA_VARIANZA*100:.0f}%")
    pca   = PCA(n_components=PCA_VARIANZA, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X.values)
    n_comp = pca.n_components_
    var_exp = pca.explained_variance_ratio_.sum() * 100
    print(f"  Componentes PCA : {n_comp}")
    print(f"  Varianza expl.  : {var_exp:.2f}%")
    print(f"  Columnas        : {X.shape[1]} → {n_comp}")

    X_pca_df = pd.DataFrame(X_pca)

    # ── Paso 2: Muestra estratificada ────────────────────────────────────
    print(f"\n  [PASO 2] Muestra estratificada — {N_MUESTRA:,} filas")
    X_muestra, indices = _muestra_estratificada(X_pca_df, df_unif)
    print(f"  Total muestra   : {len(X_muestra):,} filas")

    # ── Paso 3: t-SNE ────────────────────────────────────────────────────
    print(f"\n  [PASO 3] t-SNE")
    print(f"  Parámetros: perplexity={PERPLEXITY}, n_iter={N_ITER}, "
          f"random_state={RANDOM_STATE}")
    print(f"  Ejecutando t-SNE... (esto puede tardar ~2-4 min)")

    tsne  = TSNE(
        n_components=N_COMPONENTS,
        perplexity=PERPLEXITY,
        max_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    X_tsne = tsne.fit_transform(X_muestra.values)

    # Exportar coordenadas
    df_tsne = pd.DataFrame(X_tsne, columns=["tsne1", "tsne2"])
    df_tsne.to_csv(ARCHIVO_TSNE_2D, index=False, encoding="utf-8")

    # Exportar índices originales para alinear con unificado en el dashboard
    pd.DataFrame({"idx_original": indices}).to_csv(
        ARCHIVO_IDX_TSNE, index=False, encoding="utf-8"
    )

    print(f"\n  Filas proyectadas : {len(df_tsne):,}")
    print(f"  Exportado → {ARCHIVO_TSNE_2D}")
    print(f"  Exportado → {ARCHIVO_IDX_TSNE}")

    _grafica_rapida(df_tsne)
    return df_tsne


def cargar_tsne_2d() -> tuple:
    """Carga crime_tsne_2d.csv y crime_tsne_indices.csv."""
    if not os.path.exists(ARCHIVO_TSNE_2D):
        print(f"\n[ERROR] No existe: {ARCHIVO_TSNE_2D}")
        return pd.DataFrame(), np.array([])

    df   = pd.read_csv(ARCHIVO_TSNE_2D)
    idx  = pd.read_csv(ARCHIVO_IDX_TSNE)["idx_original"].values
    print(f"  [carga] crime_tsne_2d.csv → {len(df):,} filas")
    return df, idx


def _grafica_rapida(df_tsne: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df_tsne["tsne1"], df_tsne["tsne2"],
               s=1.5, alpha=0.4, color="#555555")
    ax.set_title(
        f"t-SNE 2D — Vista general\n"
        f"perplexity={PERPLEXITY}  |  n_iter={N_ITER}  |  "
        f"{len(df_tsne):,} puntos",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    ruta = os.path.join(RUTA_OUTPUTS, "tsne_2d_general.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")