import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RUTA_PROCESSED   = os.path.join("data", "processed")
ARCHIVO_CLUSTERS = os.path.join(RUTA_PROCESSED, "crime_clusters.csv")

K_MIN        = 2
K_MAX        = 10
RANDOM_STATE = 42
N_INIT       = 10
# Muestra para Silhouette (costoso con 473k)
N_SILHOUETTE = 20_000


def aplicar_kmeans(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica K-Means sobre el vector de características completo.
    Determina k óptimo con Silhouette Score (k entre K_MIN y K_MAX).
    Exporta crime_clusters.csv con columna 'cluster'.
    """
    os.makedirs(RUTA_PROCESSED, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  K-MEANS — CLUSTERING")
    print(f"{'='*55}")
    print(f"  Filas         : {len(df_features):,}")
    print(f"  Columnas      : {df_features.shape[1]}")
    print(f"  Rango k       : {K_MIN} a {K_MAX}")

    X = df_features.copy()

    # Rellenar nulos con media
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    # ── Silhouette Score para elegir k óptimo ─────────────────────────────
    print(f"\n  [Silhouette Score — muestra de {N_SILHOUETTE:,} filas]")
    rng     = np.random.default_rng(RANDOM_STATE)
    idx_sil = rng.choice(len(X), size=min(N_SILHOUETTE, len(X)), replace=False)
    X_sil   = X.iloc[idx_sil].values

    scores = {}
    for k in range(K_MIN, K_MAX + 1):
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=N_INIT, max_iter=300)
        labels = km.fit_predict(X_sil)
        score  = silhouette_score(X_sil, labels,
                                  sample_size=10_000,
                                  random_state=RANDOM_STATE)
        scores[k] = score
        marca = " ◄" if score == max(scores.values()) else ""
        print(f"    k={k:>2}  Silhouette={score:.4f}{marca}")

    k_optimo = max(scores, key=scores.get)
    print(f"\n  k óptimo seleccionado: {k_optimo}  "
          f"(Silhouette={scores[k_optimo]:.4f})")

    # ── K-Means final sobre toda la data ──────────────────────────────────
    print(f"\n  Aplicando K-Means (k={k_optimo}) sobre {len(X):,} filas...")
    km_final = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE,
                      n_init=N_INIT, max_iter=300)
    labels_all = km_final.fit_predict(X.values)

    # ── Exportar solo la columna cluster ──────────────────────────────────
    df_clusters = pd.DataFrame({
        "cluster": labels_all,
    })
    df_clusters.to_csv(ARCHIVO_CLUSTERS, index=False, encoding="utf-8")

    # Reporte por cluster
    print(f"\n  [DISTRIBUCIÓN POR CLUSTER]")
    print(f"  {'Cluster':<12} {'Filas':>10}  {'%':>6}")
    print(f"  {'-'*32}")
    for c in range(k_optimo):
        n   = (labels_all == c).sum()
        pct = n / len(labels_all) * 100
        print(f"  Cluster {c:<4}  {n:>10,}  {pct:>5.1f}%")

    print(f"\n  Exportado → {ARCHIVO_CLUSTERS}")
    return df_clusters


def cargar_clusters() -> pd.DataFrame:
    """Carga el CSV de clusters desde data/processed/."""
    if not os.path.exists(ARCHIVO_CLUSTERS):
        print(f"\n[ERROR] No existe: {ARCHIVO_CLUSTERS}")
        return pd.DataFrame()
    df = pd.read_csv(ARCHIVO_CLUSTERS)
    print(f"  [carga] crime_clusters.csv → {len(df):,} filas")
    return df