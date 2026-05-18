import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.translations import (
    CIUDADES_LABEL, COLORES_CIUDAD,
    CATEGORIA_LABEL, PERIODO_LABEL, ORDEN_PERIODO,
)

RUTA_OUTPUTS  = os.path.join("outputs", "clustering")
K_MIN, K_MAX  = 2, 10
RANDOM_STATE  = 42
PALETA        = [
    "#E53935","#1E88E5","#43A047","#FB8C00","#8E24AA",
    "#00ACC1","#F4511E","#3949AB","#00897B","#FFB300",
]


def ejecutar_clustering(datasets: dict) -> None:
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)

    for ciudad, df in datasets.items():
        label = CIUDADES_LABEL.get(ciudad, ciudad)
        print(f"\n{'='*60}")
        print(f"  CLUSTERING — {label.upper()}")
        print(f"{'='*60}")

        _clustering_geografico(ciudad, label, df)
        _clustering_comportamental(ciudad, label, df)

    print(f"\n  ✔ Clustering completado. Gráficas en: {RUTA_OUTPUTS}/")


# ─── Geográfico ───────────────────────────────────────────────────────────────

def _clustering_geografico(ciudad: str, label: str, df: pd.DataFrame) -> None:
    print(f"\n  [GEO] Preparando datos...")

    cols = ["latitud", "longitud"]
    df_geo = df[cols].dropna()

    if len(df_geo) < K_MAX * 10:
        print(f"  [GEO] Datos insuficientes. Omitiendo.")
        return

    # Muestra para eficiencia con datasets grandes
    muestra = df_geo.sample(min(50_000, len(df_geo)), random_state=RANDOM_STATE)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(muestra)

    k_optimo, score_optimo, scores = _elegir_k(X_scaled)
    print(f"  [GEO] k óptimo: {k_optimo}  (Silhouette={score_optimo:.4f})")

    # Ajuste final con k óptimo sobre muestra
    kmeans  = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init=10)
    labels  = kmeans.fit_predict(X_scaled)
    centros = scaler.inverse_transform(kmeans.cluster_centers_)

    # Consola
    _consola_geo(muestra, labels, k_optimo, centros, scores)

    # Gráficas
    _grafica_silhouette(scores, k_optimo, label, "geo", ciudad)
    _grafica_geo(muestra, labels, centros, k_optimo, score_optimo, label, ciudad)


def _consola_geo(muestra: pd.DataFrame, labels: np.ndarray,
                 k: int, centros: np.ndarray, scores: dict) -> None:
    print(f"\n  [GEO] Silhouette Scores por k:")
    for ki, s in scores.items():
        marca = " ◄ óptimo" if ki == max(scores, key=scores.get) else ""
        print(f"    k={ki}: {s:.4f}{marca}")

    print(f"\n  [GEO] Distribución de clusters (k={k}):")
    muestra = muestra.copy()
    muestra["cluster"] = labels
    for c in range(k):
        n = (muestra["cluster"] == c).sum()
        pct = n / len(muestra) * 100
        centro = centros[c]
        print(f"    Cluster {c+1}: {n:,} puntos ({pct:.1f}%)  "
              f"centro=({centro[0]:.4f}, {centro[1]:.4f})")


def _grafica_geo(muestra: pd.DataFrame, labels: np.ndarray, centros: np.ndarray,
                 k: int, score: float, label: str, ciudad: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for c in range(k):
        mask = labels == c
        ax.scatter(
            muestra["longitud"].values[mask],
            muestra["latitud"].values[mask],
            s=0.8, alpha=0.4, color=PALETA[c % len(PALETA)],
            label=f"Zona {c+1}",
        )

    # Centroides
    ax.scatter(
        centros[:, 1], centros[:, 0],
        s=180, marker="X", color="black", zorder=5, label="Centroide"
    )

    ax.set_title(
        f"Clustering Geográfico — {label}\n"
        f"K-Means  |  k={k} clusters  |  Silhouette={score:.4f}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(markerscale=6, fontsize=9, title="Zona geográfica")
    plt.tight_layout()
    _guardar(fig, f"clustering_geo_{ciudad}.png")


# ─── Comportamental ───────────────────────────────────────────────────────────

def _clustering_comportamental(ciudad: str, label: str, df: pd.DataFrame) -> None:
    print(f"\n  [COMP] Preparando datos...")

    # Encoding de categoria_delito
    if "categoria_delito" not in df.columns:
        print(f"  [COMP] Sin categoria_delito. Omitiendo.")
        return

    df_comp = df[["hora", "mes", "es_fin_semana", "es_peligroso", "categoria_delito"]].dropna()
    df_comp = df_comp.copy()
    df_comp["es_fin_semana"] = df_comp["es_fin_semana"].astype(int)
    df_comp["es_peligroso"]  = df_comp["es_peligroso"].astype(int)

    # One-hot encoding categoria_delito
    dummies   = pd.get_dummies(df_comp["categoria_delito"], prefix="cat")
    df_comp   = pd.concat([df_comp.drop(columns=["categoria_delito"]), dummies], axis=1)

    muestra   = df_comp.sample(min(50_000, len(df_comp)), random_state=RANDOM_STATE)
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(muestra)

    k_optimo, score_optimo, scores = _elegir_k(X_scaled)
    print(f"  [COMP] k óptimo: {k_optimo}  (Silhouette={score_optimo:.4f})")

    kmeans = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    muestra = muestra.copy()
    muestra["cluster"] = labels

    _consola_comp(muestra, labels, k_optimo, scores)
    _grafica_silhouette(scores, k_optimo, label, "comp", ciudad)
    _grafica_comp_pca(X_scaled, labels, k_optimo, score_optimo, label, ciudad)
    _grafica_comp_perfil(muestra, k_optimo, label, ciudad)


def _consola_comp(muestra: pd.DataFrame, labels: np.ndarray,
                  k: int, scores: dict) -> None:
    print(f"\n  [COMP] Silhouette Scores por k:")
    for ki, s in scores.items():
        marca = " ◄ óptimo" if ki == max(scores, key=scores.get) else ""
        print(f"    k={ki}: {s:.4f}{marca}")

    print(f"\n  [COMP] Perfil de clusters (k={k}):")
    for c in range(k):
        sub = muestra[muestra["cluster"] == c]
        n   = len(sub)
        pct = n / len(muestra) * 100
        print(f"\n    Cluster {c+1}: {n:,} registros ({pct:.1f}%)")
        print(f"      Hora media      : {sub['hora'].mean():.1f}h")
        print(f"      Mes medio       : {sub['mes'].mean():.1f}")
        print(f"      Fin de semana   : {sub['es_fin_semana'].mean()*100:.1f}%")
        print(f"      Peligrosos      : {sub['es_peligroso'].mean()*100:.1f}%")


def _grafica_comp_pca(X_scaled: np.ndarray, labels: np.ndarray,
                      k: int, score: float, label: str, ciudad: str) -> None:
    """Proyección PCA 2D para visualizar clusters comportamentales."""
    pca  = PCA(n_components=2, random_state=RANDOM_STATE)
    X2d  = pca.fit_transform(X_scaled)
    var  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(k):
        mask = labels == c
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   s=1.5, alpha=0.4, color=PALETA[c % len(PALETA)],
                   label=f"Perfil {c+1}")

    ax.set_title(
        f"Clustering Comportamental — {label}  (PCA 2D)\n"
        f"K-Means  |  k={k} clusters  |  Silhouette={score:.4f}  |  "
        f"Varianza explicada: {var[0]:.1f}% + {var[1]:.1f}%",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel(f"Componente Principal 1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"Componente Principal 2 ({var[1]:.1f}%)")
    ax.legend(markerscale=8, fontsize=9, title="Perfil delictivo")
    plt.tight_layout()
    _guardar(fig, f"clustering_comp_pca_{ciudad}.png")


def _grafica_comp_perfil(muestra: pd.DataFrame, k: int,
                         label: str, ciudad: str) -> None:
    """Heatmap de características medias por cluster."""
    cols_perfil = ["hora", "mes", "es_fin_semana", "es_peligroso"]
    cols_perfil = [c for c in cols_perfil if c in muestra.columns]

    perfil = muestra.groupby("cluster")[cols_perfil].mean()
    perfil.index = [f"Perfil {c+1}" for c in perfil.index]

    LABELS_COLS = {
        "hora":          "Hora media",
        "mes":           "Mes medio",
        "es_fin_semana": "% Fin semana",
        "es_peligroso":  "% Peligroso",
    }
    perfil.columns = [LABELS_COLS.get(c, c) for c in perfil.columns]

    # Normalizar para heatmap comparable
    perfil_norm = (perfil - perfil.min()) / (perfil.max() - perfil.min() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, k)))

    # Heatmap valores normalizados
    sns.heatmap(perfil_norm, annot=perfil.round(2), fmt="g",
                cmap="YlOrRd", linewidths=0.5,
                ax=axes[0], cbar_kws={"label": "Valor normalizado"})
    axes[0].set_title("Características por perfil (normalizado)")
    axes[0].set_xlabel("")

    # Barras: % peligroso por cluster
    pct_pel = muestra.groupby("cluster")["es_peligroso"].mean() * 100
    colores = [PALETA[c % len(PALETA)] for c in range(k)]
    axes[1].bar([f"Perfil {c+1}" for c in range(k)],
                pct_pel.values, color=colores, edgecolor="white")
    axes[1].set_title("% Delitos peligrosos por perfil")
    axes[1].set_ylabel("% Peligrosos")
    axes[1].set_ylim(0, max(pct_pel.max() * 1.2, 10))
    for i, v in enumerate(pct_pel.values):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    fig.suptitle(
        f"Perfiles delictivos — {label}  |  k={k} clusters",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _guardar(fig, f"clustering_comp_perfil_{ciudad}.png")


# ─── Silhouette plot ──────────────────────────────────────────────────────────

def _grafica_silhouette(scores: dict, k_optimo: int,
                        label: str, tipo: str, ciudad: str) -> None:
    tipo_label = "Geográfico" if tipo == "geo" else "Comportamental"
    ks     = list(scores.keys())
    vals   = list(scores.values())
    colores = ["#EF5350" if k == k_optimo else "#90CAF9" for k in ks]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ks, vals, color=colores, edgecolor="white", width=0.6)
    ax.set_title(
        f"Silhouette Score por número de clusters\n"
        f"Clustering {tipo_label} — {label}  |  k óptimo={k_optimo}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(ks)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_ylim(0, max(vals) * 1.2)

    parches = [
        mpatches.Patch(color="#EF5350", label=f"k óptimo ({k_optimo})"),
        mpatches.Patch(color="#90CAF9", label="Otros k"),
    ]
    ax.legend(handles=parches)
    plt.tight_layout()
    _guardar(fig, f"clustering_silhouette_{tipo}_{ciudad}.png")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _elegir_k(X_scaled: np.ndarray) -> tuple[int, float, dict]:
    """Evalúa k de K_MIN a K_MAX y retorna el k con mayor Silhouette Score."""
    scores = {}
    for k in range(K_MIN, K_MAX + 1):
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labs   = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labs, sample_size=10_000,
                                     random_state=RANDOM_STATE)
        print(f"    k={k} → Silhouette={scores[k]:.4f}")

    k_optimo     = max(scores, key=scores.get)
    score_optimo = scores[k_optimo]
    return k_optimo, score_optimo, scores


def _guardar(fig: plt.Figure, nombre: str) -> None:
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")