import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RUTA_PROCESSED  = os.path.join("data", "processed")
RUTA_OUTPUTS    = os.path.join("outputs", "pca")
ARCHIVO_PCA_2D  = os.path.join(RUTA_PROCESSED, "crime_pca_2d.csv")
ARCHIVO_PCA_3D  = os.path.join(RUTA_PROCESSED, "crime_pca_3d.csv")

# Columnas que NO entran al PCA
COLS_EXCLUIR = ["anio", "latitud", "longitud"]


def aplicar_pca(df_features: pd.DataFrame, df_unificado: pd.DataFrame) -> tuple:
    """
    Aplica PCA reduciendo a 2D y 3D.
    Conserva latitud, longitud, anio y columnas originales fuera del PCA.
    Retorna (df_pca_2d, df_pca_3d)
    """
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    os.makedirs(RUTA_PROCESSED, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  PCA — REDUCCIÓN A 2D y 3D")
    print(f"{'='*50}")

    # Columnas que entran al PCA
    cols_excluir = [c for c in COLS_EXCLUIR if c in df_features.columns]
    X = df_features.drop(columns=cols_excluir).copy()

    # Eliminar columnas con varianza 0
    cols_var_cero = X.columns[X.std() == 0].tolist()
    if cols_var_cero:
        X = X.drop(columns=cols_var_cero)
        print(f"  [info] Columnas varianza 0 omitidas: {cols_var_cero}")

    print(f"  Columnas entrada : {X.shape[1]}")
    print(f"  Filas            : {len(X):,}")

    # ── PCA 2D ────────────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=42)
    X2d  = pca2.fit_transform(X)
    var2 = pca2.explained_variance_ratio_ * 100
    print(f"\n  [2D] Varianza explicada: PC1={var2[0]:.2f}%  PC2={var2[1]:.2f}%  "
          f"Total={sum(var2):.2f}%")

    # ── PCA 3D ────────────────────────────────────────────────
    pca3 = PCA(n_components=3, random_state=42)
    X3d  = pca3.fit_transform(X)
    var3 = pca3.explained_variance_ratio_ * 100
    print(f"  [3D] Varianza explicada: PC1={var3[0]:.2f}%  PC2={var3[1]:.2f}%  "
          f"PC3={var3[2]:.2f}%  Total={sum(var3):.2f}%")

    # Alinear con unificado para columnas originales
    n = min(len(X2d), len(df_unificado))
    meta = df_unificado[["ciudad", "categoria_delito", "es_peligroso",
                          "hora", "mes", "fecha", "latitud", "longitud"]].iloc[:n].reset_index(drop=True)

    # ── DataFrames resultado ──────────────────────────────────
    df_2d = pd.DataFrame(X2d[:n], columns=["pc1", "pc2"])
    df_2d = pd.concat([df_2d, meta], axis=1)

    df_3d = pd.DataFrame(X3d[:n], columns=["pc1", "pc2", "pc3"])
    df_3d = pd.concat([df_3d, meta], axis=1)

    df_2d.to_csv(ARCHIVO_PCA_2D, index=False, encoding="utf-8")
    df_3d.to_csv(ARCHIVO_PCA_3D, index=False, encoding="utf-8")
    print(f"\n  Exportado → {ARCHIVO_PCA_2D}")
    print(f"  Exportado → {ARCHIVO_PCA_3D}")

    # Gráfica de varianza informativa
    _grafica_varianza_comparativa(pca2, pca3, X.shape[1])

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

def _grafica_varianza_comparativa(pca2: PCA, pca3: PCA, n_cols: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    categorias = ["2D\n(PC1+PC2)", "3D\n(PC1+PC2+PC3)"]
    valores    = [
        sum(pca2.explained_variance_ratio_) * 100,
        sum(pca3.explained_variance_ratio_) * 100,
    ]
    colores = ["#2196F3", "#4CAF50"]
    bars = ax.bar(categorias, valores, color=colores, edgecolor="white", width=0.4)
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