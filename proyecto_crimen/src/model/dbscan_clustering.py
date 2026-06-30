"""
DBSCAN clustering sobre la proyección 2D (t-SNE o UMAP).
A diferencia de K-Means, DBSCAN no requiere definir k de antemano
y detecta automáticamente outliers (ruido) con etiqueta -1.

Parámetros clave:
    eps         : radio máximo entre vecinos para pertenecer al mismo cluster
    min_samples : mínimo de puntos para formar un cluster denso
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

RUTA_PROCESSED      = os.path.join("data", "processed")
RUTA_OUTPUTS        = os.path.join("outputs", "dbscan")
ARCHIVO_TSNE        = os.path.join(RUTA_PROCESSED, "crime_tsne_2d.csv")
ARCHIVO_TSNE_IDX    = os.path.join(RUTA_PROCESSED, "crime_tsne_indices.csv")
ARCHIVO_UMAP        = os.path.join(RUTA_PROCESSED, "crime_umap_2d.csv")
ARCHIVO_DBSCAN      = os.path.join(RUTA_PROCESSED, "crime_dbscan.csv")

# ─── Parámetros DBSCAN ───────────────────────────────────────────────────────
MIN_SAMPLES  = 5      # mínimo de puntos para formar un cluster
EPS_AUTO     = True   # True = calcular eps automáticamente con KNN
EPS_MANUAL   = 1.5    # valor usado solo si EPS_AUTO = False
FUENTE       = "tsne" # "tsne" o "umap"


def _calcular_eps_knn(X: np.ndarray) -> float:
    """
    Calcula eps óptimo usando la curva de distancias KNN.
    El codo de la curva indica el eps adecuado.
    """
    print(f"  Calculando curva KNN (k={MIN_SAMPLES})...")
    nbrs    = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
    distancias, _ = nbrs.kneighbors(X)
    distancias_sorted = np.sort(distancias[:, -1])

    # Guardar gráfica de la curva KNN
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distancias_sorted, linewidth=1.5, color="#2196F3")
    ax.set_title(f"Curva KNN — distancia al {MIN_SAMPLES}° vecino más cercano\n"
                 f"El codo indica el eps óptimo para DBSCAN", fontsize=12)
    ax.set_xlabel("Puntos ordenados")
    ax.set_ylabel(f"Distancia al {MIN_SAMPLES}° vecino")
    ax.grid(True, alpha=0.3)
    ruta_knn = os.path.join(RUTA_OUTPUTS, "knn_curve.png")
    fig.savefig(ruta_knn, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Curva KNN guardada → {ruta_knn}")

    # Detectar el codo automáticamente usando la segunda derivada
    segunda_derivada = np.diff(distancias_sorted, n=2)
    idx_codo = np.argmax(segunda_derivada) + 2
    eps_optimo = distancias_sorted[idx_codo]

    print(f"  Índice del codo detectado : {idx_codo}")
    print(f"  eps automático            : {eps_optimo:.4f}")
    return float(eps_optimo)


def _cargar_proyeccion() -> tuple:
    """Carga la proyección 2D según FUENTE (tsne o umap)."""
    if FUENTE == "tsne":
        if not os.path.exists(ARCHIVO_TSNE):
            print(f"[ERROR] No existe: {ARCHIVO_TSNE}")
            return None, None
        df = pd.read_csv(ARCHIVO_TSNE)
        idx = None
        if os.path.exists(ARCHIVO_TSNE_IDX):
            idx = pd.read_csv(ARCHIVO_TSNE_IDX)["idx_original"].values
        print(f"  [carga] crime_tsne_2d.csv → {len(df):,} filas")
        return df, idx

    elif FUENTE == "umap":
        if not os.path.exists(ARCHIVO_UMAP):
            print(f"[ERROR] No existe: {ARCHIVO_UMAP}")
            return None, None
        df = pd.read_csv(ARCHIVO_UMAP)
        print(f"  [carga] crime_umap_2d.csv → {len(df):,} filas")
        return df, None

    else:
        print(f"[ERROR] FUENTE inválida: {FUENTE}. Usa 'tsne' o 'umap'")
        return None, None


def aplicar_dbscan() -> pd.DataFrame:
    """
    Aplica DBSCAN sobre la proyección 2D seleccionada.
    Exporta crime_dbscan.csv con columnas:
        idx_original  : índice en crime_unified.csv
        dbscan_label  : cluster asignado (-1 = ruido/outlier)
    """
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    os.makedirs(RUTA_PROCESSED, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  DBSCAN — CLUSTERING SOBRE PROYECCIÓN 2D")
    print(f"{'='*55}")
    print(f"  Fuente          : {FUENTE.upper()}")
    print(f"  min_samples     : {MIN_SAMPLES}")
    print(f"  eps_auto        : {EPS_AUTO}")

    # Cargar proyección
    df_proj, idx_original = _cargar_proyeccion()
    if df_proj is None:
        return pd.DataFrame()

    # Determinar nombres de columnas
    col_x = "tsne1" if FUENTE == "tsne" else "umap1"
    col_y = "tsne2" if FUENTE == "tsne" else "umap2"
    X = df_proj[[col_x, col_y]].values

    # Calcular eps
    if EPS_AUTO:
        eps = _calcular_eps_knn(X)
    else:
        eps = EPS_MANUAL
        print(f"  eps manual      : {eps}")

    # Aplicar DBSCAN
    print(f"\n  Aplicando DBSCAN sobre {len(X):,} puntos...")
    print(f"  eps={eps:.4f}, min_samples={MIN_SAMPLES}")
    db      = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, n_jobs=-1)
    labels  = db.fit_predict(X)

    # Estadísticas
    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido     = (labels == -1).sum()
    pct_ruido   = n_ruido / len(labels) * 100

    print(f"\n  [RESULTADOS]")
    print(f"  Clusters encontrados : {n_clusters}")
    print(f"  Puntos como ruido    : {n_ruido:,} ({pct_ruido:.1f}%)")
    print(f"  Puntos en clusters   : {len(labels) - n_ruido:,} ({100-pct_ruido:.1f}%)")

    print(f"\n  [DISTRIBUCIÓN POR CLUSTER]")
    print(f"  {'Cluster':<12} {'Puntos':>10} {'%':>8}")
    print(f"  {'-'*33}")
    for label in sorted(set(labels)):
        n   = (labels == label).sum()
        pct = n / len(labels) * 100
        nombre = f"Ruido (-1)" if label == -1 else f"Cluster {label}"
        print(f"  {nombre:<12} {n:>10,} {pct:>8.2f}%")

    # Construir DataFrame de salida
    if idx_original is not None:
        df_out = pd.DataFrame({
            "idx_original": idx_original,
            "dbscan_label": labels,
        })
    else:
        df_out = pd.DataFrame({
            "idx_original": np.arange(len(labels)),
            "dbscan_label": labels,
        })

    df_out.to_csv(ARCHIVO_DBSCAN, index=False, encoding="utf-8")
    print(f"\n  Exportado → {ARCHIVO_DBSCAN}")

    _grafica_rapida(X, labels, n_clusters)
    return df_out


def _grafica_rapida(X: np.ndarray, labels: np.ndarray,
                    n_clusters: int) -> None:
    """Gráfica rápida de los clusters DBSCAN."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colores_unicos = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    for i, label in enumerate(sorted(set(labels))):
        mask  = labels == label
        color = "#cccccc" if label == -1 else colores_unicos[i % len(colores_unicos)]
        nombre = "Ruido" if label == -1 else f"Cluster {label}"
        ax.scatter(X[mask, 0], X[mask, 1], s=1.5, alpha=0.4,
                   color=color, label=nombre)

    ax.set_title(
        f"DBSCAN sobre {FUENTE.upper()} 2D\n"
        f"eps={EPS_MANUAL if not EPS_AUTO else 'auto'} | "
        f"min_samples={MIN_SAMPLES} | "
        f"{n_clusters} clusters + ruido",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel(f"{FUENTE.upper()} 1")
    ax.set_ylabel(f"{FUENTE.upper()} 2")
    if n_clusters <= 15:
        ax.legend(markerscale=5, fontsize=8, loc="upper right")
    plt.tight_layout()

    ruta = os.path.join(RUTA_OUTPUTS, f"dbscan_{FUENTE}_general.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")


def cargar_dbscan() -> pd.DataFrame:
    """Carga crime_dbscan.csv desde data/processed/."""
    if not os.path.exists(ARCHIVO_DBSCAN):
        print(f"\n[ERROR] No existe: {ARCHIVO_DBSCAN}")
        return pd.DataFrame()
    df = pd.read_csv(ARCHIVO_DBSCAN)
    n_clusters = df["dbscan_label"].nunique() - (1 if -1 in df["dbscan_label"].values else 0)
    n_ruido    = (df["dbscan_label"] == -1).sum()
    print(f"  [carga] crime_dbscan.csv → {len(df):,} filas | "
          f"{n_clusters} clusters | {n_ruido:,} ruido")
    return df