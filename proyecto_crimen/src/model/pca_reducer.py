import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RUTA_PROCESSED = os.path.join("data", "processed")
RUTA_OUTPUTS   = os.path.join("outputs", "pca")
ARCHIVO_PCA    = os.path.join(RUTA_PROCESSED, "crime_pca.csv")

VARIANZA_OBJETIVO = 0.95


def aplicar_pca(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica PCA al vector de características.
    Conserva anio separado (no entra al PCA).
    Exporta crime_pca.csv y genera gráfica de varianza acumulada.
    """
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    os.makedirs(RUTA_PROCESSED, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  PCA — REDUCCIÓN DE DIMENSIONALIDAD")
    print(f"{'='*50}")

    # Separar anio antes del PCA (no aporta varianza con un solo año)
    anio_col = None
    if "anio" in df_features.columns:
        anio_col = df_features["anio"].copy()

    cols_excluir = [c for c in ["anio"] if c in df_features.columns]
    X = df_features.drop(columns=cols_excluir).copy()

    # Eliminar columnas con varianza 0 (evita warnings)
    cols_var_cero = X.columns[X.std() == 0].tolist()
    if cols_var_cero:
        X = X.drop(columns=cols_var_cero)
        print(f"  [info] Columnas con varianza 0 omitidas: {cols_var_cero}")

    print(f"  Columnas entrada : {X.shape[1]}")
    print(f"  Filas            : {len(X):,}")
    print(f"  Varianza objetivo: {VARIANZA_OBJETIVO*100:.0f}%")

    # PCA automático con varianza objetivo
    pca = PCA(n_components=VARIANZA_OBJETIVO, random_state=42)
    X_pca = pca.fit_transform(X)

    n_componentes      = pca.n_components_
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)

    print(f"\n  Componentes resultantes : {n_componentes}")
    print(f"  Varianza total explicada: {varianza_acumulada[-1]*100:.2f}%")
    print(f"  Reducción               : {X.shape[1]} → {n_componentes} columnas "
          f"({(1 - n_componentes/X.shape[1])*100:.1f}% menos)")

    # Consola: varianza por componente
    print(f"\n  [VARIANZA POR COMPONENTE]")
    print(f"  {'Componente':<15} {'Varianza':>10} {'Acumulada':>12}")
    print(f"  {'-'*40}")
    for i, (v, va) in enumerate(zip(varianza_explicada, varianza_acumulada)):
        print(f"  PC{i+1:<13} {v*100:>9.2f}%  {va*100:>10.2f}%")

    # Construir DataFrame resultado
    cols_pca = [f"pc{i+1}" for i in range(n_componentes)]
    df_pca   = pd.DataFrame(X_pca, columns=cols_pca)

    # Reagregar anio
    if anio_col is not None:
        df_pca.insert(0, "anio", anio_col.values)

    df_pca.to_csv(ARCHIVO_PCA, index=False, encoding="utf-8")
    print(f"\n  Exportado → {ARCHIVO_PCA}")

    # Gráficas
    _grafica_varianza(varianza_explicada, varianza_acumulada, n_componentes)
    _grafica_componentes_top(pca, X.columns, n_componentes)

    return df_pca


def cargar_pca() -> pd.DataFrame:
    """Carga el CSV con componentes PCA desde data/processed/."""
    if not os.path.exists(ARCHIVO_PCA):
        print(f"\n[ERROR] No existe: {ARCHIVO_PCA}")
        return pd.DataFrame()

    df = pd.read_csv(ARCHIVO_PCA, low_memory=False)
    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    print(f"  [carga] crime_pca.csv → {len(df):,} filas, {df.shape[1]} columnas")
    return df


# ─── Gráficas ─────────────────────────────────────────────────────────────────

def _grafica_varianza(varianza_explicada: np.ndarray,
                      varianza_acumulada: np.ndarray,
                      n_componentes: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PCA — Análisis de varianza explicada",
                 fontsize=13, fontweight="bold")

    componentes = range(1, len(varianza_explicada) + 1)

    # Barras: varianza por componente
    axes[0].bar(componentes, varianza_explicada * 100,
                color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_title("Varianza explicada por componente")
    axes[0].set_xlabel("Componente principal")
    axes[0].set_ylabel("Varianza explicada (%)")

    # Línea: varianza acumulada
    axes[1].plot(componentes, varianza_acumulada * 100,
                 marker="o", color="#E53935", linewidth=2.5)
    axes[1].axhline(y=95, color="gray", linestyle="--", linewidth=1.2,
                    label="95% objetivo")
    axes[1].axvline(x=n_componentes, color="#FB8C00", linestyle="--",
                    linewidth=1.2, label=f"k={n_componentes} componentes")
    axes[1].fill_between(componentes, varianza_acumulada * 100,
                         alpha=0.1, color="#E53935")
    axes[1].set_title("Varianza acumulada")
    axes[1].set_xlabel("Número de componentes")
    axes[1].set_ylabel("Varianza acumulada (%)")
    axes[1].legend()
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    _guardar(fig, "pca_varianza.png")


def _grafica_componentes_top(pca: PCA, feature_names: pd.Index,
                              n_componentes: int) -> None:
    """Heatmap de componentes vs features más importantes.
    Si hay más de 8 componentes se divide en 2 gráficas."""

    loadings = pd.DataFrame(
        pca.components_[:n_componentes],
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(n_componentes)],
    )

    # Top 15 features con mayor carga absoluta promedio
    top_features = loadings.abs().mean().nlargest(15).index
    loadings_top = loadings[top_features]

    if n_componentes <= 8:
        _heatmap_loadings(loadings_top, "pca_loadings.png",
                          f"PCA — Cargas de los {n_componentes} componentes\n"
                          f"(Top 15 variables más influyentes)")
    else:
        mitad = n_componentes // 2
        # Gráfica 1: primera mitad
        _heatmap_loadings(loadings_top.iloc[:mitad], "pca_loadings_1.png",
                          f"PCA — Cargas PC1 a PC{mitad}\n"
                          f"(Top 15 variables más influyentes)")
        # Gráfica 2: segunda mitad (incluye PC extra si impar)
        _heatmap_loadings(loadings_top.iloc[mitad:], "pca_loadings_2.png",
                          f"PCA — Cargas PC{mitad+1} a PC{n_componentes}\n"
                          f"(Top 15 variables más influyentes)")


def _heatmap_loadings(loadings: pd.DataFrame, nombre: str, titulo: str) -> None:
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(14, max(4, len(loadings) * 0.6)))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax,
                cbar_kws={"label": "Carga (loading)"})
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("Variable original")
    ax.set_ylabel("Componente principal")
    plt.tight_layout()
    _guardar(fig, nombre)


def _guardar(fig: plt.Figure, nombre: str) -> None:
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")