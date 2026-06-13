import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RUTA_PROCESSED  = os.path.join("data", "processed")
RUTA_OUTPUTS    = os.path.join("outputs", "pca")
ARCHIVO_PCA_2D  = os.path.join(RUTA_PROCESSED, "crime_pca_2d.csv")
ARCHIVO_PCA_3D  = os.path.join(RUTA_PROCESSED, "crime_pca_3d.csv")

# Solo estas columnas entran al PCA — mayoría continuas, mínimo binario
COLS_PCA = [
    "hora_scaled",
    "mes_scaled",
    "temperatura_scaled",
    "precipitacion_scaled",
    "viento_scaled",
    "es_peligroso",
    "ciudad_chicago",
    "ciudad_philadelphia",
    "ciudad_san_francisco",
]


def aplicar_pca(df_features: pd.DataFrame, df_unificado: pd.DataFrame) -> tuple:
    """
    Aplica PCA reduciendo a 2D y 3D usando solo variables continuas + ciudad OHE.
    Los CSVs resultantes contienen SOLO los componentes (pc1, pc2, pc3).
    La alineación con datos reales se hace por índice de fila en pca_viz.py.
    """
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    os.makedirs(RUTA_PROCESSED, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  PCA — REDUCCIÓN A 2D y 3D")
    print(f"{'='*50}")

    # Seleccionar solo las columnas definidas para PCA
    cols_disponibles = [c for c in COLS_PCA if c in df_features.columns]
    cols_faltantes   = [c for c in COLS_PCA if c not in df_features.columns]
    if cols_faltantes:
        print(f"  [AVISO] Columnas no encontradas: {cols_faltantes}")

    X = df_features[cols_disponibles].copy()

    # Rellenar nulos con media
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())
        print(f"  [info] Nulos rellenados con media")

    # Eliminar columnas con varianza 0
    cols_var_cero = X.columns[X.std() == 0].tolist()
    if cols_var_cero:
        X = X.drop(columns=cols_var_cero)
        print(f"  [info] Columnas varianza 0 omitidas: {cols_var_cero}")

    print(f"  Columnas entrada : {X.shape[1]}  {list(X.columns)}")
    print(f"  Filas            : {len(X):,}")

    # ── PCA 2D ────────────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=42)
    X2d  = pca2.fit_transform(X)
    var2 = pca2.explained_variance_ratio_ * 100
    print(f"\n  [2D] PC1={var2[0]:.2f}%  PC2={var2[1]:.2f}%  Total={sum(var2):.2f}%")

    # ── PCA 3D ────────────────────────────────────────────────
    pca3 = PCA(n_components=3, random_state=42)
    X3d  = pca3.fit_transform(X)
    var3 = pca3.explained_variance_ratio_ * 100
    print(f"  [3D] PC1={var3[0]:.2f}%  PC2={var3[1]:.2f}%  PC3={var3[2]:.2f}%  Total={sum(var3):.2f}%")

    # ── Exportar solo componentes ──────────────────────────────
    df_2d = pd.DataFrame(X2d, columns=["pc1", "pc2"])
    df_3d = pd.DataFrame(X3d, columns=["pc1", "pc2", "pc3"])

    df_2d.to_csv(ARCHIVO_PCA_2D, index=False, encoding="utf-8")
    df_3d.to_csv(ARCHIVO_PCA_3D, index=False, encoding="utf-8")
    print(f"\n  Exportado → {ARCHIVO_PCA_2D}  ({df_2d.shape[1]} cols)")
    print(f"  Exportado → {ARCHIVO_PCA_3D}  ({df_3d.shape[1]} cols)")

    # Gráfica informativa
    _grafica_varianza(pca2, pca3, X.shape[1])

    return df_2d, df_3d


def cargar_pca_2d() -> pd.DataFrame:
    return _cargar(ARCHIVO_PCA_2D, "crime_pca_2d.csv")


def cargar_pca_3d() -> pd.DataFrame:
    return _cargar(ARCHIVO_PCA_3D, "crime_pca_3d.csv")


def _cargar(ruta: str, nombre: str) -> pd.DataFrame:
    if not os.path.exists(ruta):
        print(f"\n[ERROR] No existe: {ruta}")
        return pd.DataFrame()
    df = pd.read_csv(ruta, low_memory=False)
    print(f"  [carga] {nombre} → {len(df):,} filas, {df.shape[1]} columnas")
    return df


# ─── Gráfica informativa ──────────────────────────────────────────────────────

def _grafica_varianza(pca2: PCA, pca3: PCA, n_cols: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    categorias = ["2D\n(PC1+PC2)", "3D\n(PC1+PC2+PC3)"]
    valores    = [
        sum(pca2.explained_variance_ratio_) * 100,
        sum(pca3.explained_variance_ratio_) * 100,
    ]
    bars = ax.bar(categorias, valores,
                  color=["#2196F3", "#4CAF50"], edgecolor="white", width=0.4)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11)
    ax.set_title(
        f"Varianza explicada por reducción PCA\n"
        f"(desde {n_cols} dimensiones originales)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_ylim(0, 100)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()

    ruta = os.path.join(RUTA_OUTPUTS, "pca_varianza_2d_3d.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")