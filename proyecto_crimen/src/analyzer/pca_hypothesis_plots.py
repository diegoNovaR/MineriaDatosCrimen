"""
Genera 5 gráficas de dispersión PCA para análisis de hipótesis de crimen urbano.
Salida: outputs/pca/*.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ─── Rutas ───────────────────────────────────────────────────────────────────
RUTA_PCA     = os.path.join("data", "processed", "crime_pca.csv")
RUTA_UNIFIED = os.path.join("data", "processed", "crime_unified.csv")
RUTA_OUTPUT  = os.path.join("outputs", "pca")
os.makedirs(RUTA_OUTPUT, exist_ok=True)

# ─── Paletas ─────────────────────────────────────────────────────────────────
COLORES_CIUDAD = {
    "chicago":       "#2196F3",   # azul
    "philadelphia":  "#FF9800",   # naranja
    "san_francisco": "#4CAF50",   # verde
}
COLOR_PELIGROSO     = "#E53935"   # rojo
COLOR_NO_PELIGROSO  = "#90CAF9"   # azul claro
COLOR_DEFAULT       = "#607D8B"   # gris

ALPHA = 0.45
SIZE  = 8


# ─── Carga ───────────────────────────────────────────────────────────────────
def cargar_datos():
    print("[carga] Leyendo crime_pca.csv ...")
    df_pca = pd.read_csv(RUTA_PCA, low_memory=False)

    # Intentar enriquecer con ciudad y es_peligroso del unificado
    df_meta = None
    if os.path.exists(RUTA_UNIFIED):
        print("[carga] Leyendo crime_unified.csv para metadatos ...")
        df_meta = pd.read_csv(
            RUTA_UNIFIED,
            usecols=["ciudad", "es_peligroso"],
            low_memory=False
        )
        if len(df_meta) == len(df_pca):
            df_pca = pd.concat([df_pca.reset_index(drop=True),
                                 df_meta.reset_index(drop=True)], axis=1)
            print(f"  ✔ Metadatos unidos ({len(df_pca):,} filas)")
        else:
            print(f"  [AVISO] Longitudes distintas — PCA:{len(df_pca)}, "
                  f"Unified:{len(df_meta)}. Se omiten metadatos.")
            df_meta = None

    if df_meta is None:
        df_pca["ciudad"]       = "sin_datos"
        df_pca["es_peligroso"] = False

    # Normalizar es_peligroso
    df_pca["es_peligroso"] = df_pca["es_peligroso"].astype(str).str.lower()
    df_pca["es_peligroso"] = df_pca["es_peligroso"].map(
        {"true": True, "1": True, "false": False, "0": False}
    ).fillna(False)

    print(f"  Columnas disponibles: {list(df_pca.columns)}")
    return df_pca


# ─── Helpers de graficado ─────────────────────────────────────────────────────
def _scatter_ciudad(ax, df, xcol, ycol):
    """Scatter coloreado por ciudad."""
    ciudades = sorted(df["ciudad"].unique())
    for ciudad in ciudades:
        mask = df["ciudad"] == ciudad
        color = COLORES_CIUDAD.get(ciudad, COLOR_DEFAULT)
        ax.scatter(
            df.loc[mask, xcol], df.loc[mask, ycol],
            c=color, s=SIZE, alpha=ALPHA, linewidths=0, label=ciudad.replace("_", " ").title()
        )


def _scatter_peligrosidad(ax, df, xcol, ycol):
    """Scatter coloreado por peligrosidad."""
    for peligroso, color, etiqueta in [
        (True,  COLOR_PELIGROSO,    "Peligroso"),
        (False, COLOR_NO_PELIGROSO, "No peligroso"),
    ]:
        mask = df["es_peligroso"] == peligroso
        ax.scatter(
            df.loc[mask, xcol], df.loc[mask, ycol],
            c=color, s=SIZE, alpha=ALPHA, linewidths=0, label=etiqueta
        )


def _estilo_ax(ax, xlabel, ylabel, titulo):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(markerscale=2.5, fontsize=9, framealpha=0.7,
              loc="upper right", borderpad=0.6)


def _guardar(fig, nombre, descripcion):
    ruta = os.path.join(RUTA_OUTPUT, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [✔] Guardada → {ruta}  ({descripcion})")


# ─── 5 Gráficas ──────────────────────────────────────────────────────────────

def plot_pc1_pc2(df):
    """PC1 × PC2 — Visión general: geografía vs tiempo."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle("PC1 × PC2 — Visión general: geografía vs tiempo",
                 fontsize=14, fontweight="bold", y=1.01)

    # Panel izq: por ciudad
    _scatter_ciudad(axes[0], df, "pc1", "pc2")
    _estilo_ax(axes[0], "PC1 (posición geográfica)", "PC2 (hora del delito)",
               "Por ciudad")

    # Panel der: por peligrosidad
    _scatter_peligrosidad(axes[1], df, "pc1", "pc2")
    _estilo_ax(axes[1], "PC1 (posición geográfica)", "PC2 (hora del delito)",
               "Por peligrosidad")

    _agregar_nota(fig,
        "PC1 captura la variación geográfica principal (lat/lon + ciudad).\n"
        "PC2 refleja el patrón temporal por hora del día.")

    plt.tight_layout()
    _guardar(fig, "pca_pc1_pc2_vision_general.png", "visión general geografía vs tiempo")


def plot_h1_pc2_pc3(df):
    """H1 — Patrones temporales: PC2 × PC3 (hora × mes)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("H1 — Patrones temporales: PC2 (hora) × PC3 (mes)",
                 fontsize=14, fontweight="bold", y=1.01)

    _scatter_ciudad(axes[0], df, "pc2", "pc3")
    _estilo_ax(axes[0], "PC2 (hora del delito)", "PC3 (mes / estacionalidad)",
               "Por ciudad")

    _scatter_peligrosidad(axes[1], df, "pc2", "pc3")
    _estilo_ax(axes[1], "PC2 (hora del delito)", "PC3 (mes / estacionalidad)",
               "Por peligrosidad")

    _agregar_nota(fig,
        "H1: Los delitos presentan patrones temporales distintos según hora y mes.\n"
        "Agrupaciones visibles sugieren horarios y estaciones de mayor incidencia.")

    plt.tight_layout()
    _guardar(fig, "pca_h1_patrones_temporales.png", "H1 patrones temporales")


def plot_h2_pc1_pc4(df):
    """H2 — Concentración geográfica: PC1 × PC4."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("H2 — Concentración geográfica: PC1 × PC4",
                 fontsize=14, fontweight="bold", y=1.01)

    _scatter_ciudad(axes[0], df, "pc1", "pc4")
    _estilo_ax(axes[0], "PC1 (latitud/longitud principal)", "PC4 (lat/lon + ciudad)",
               "Por ciudad")

    _scatter_peligrosidad(axes[1], df, "pc1", "pc4")
    _estilo_ax(axes[1], "PC1 (latitud/longitud principal)", "PC4 (lat/lon + ciudad)",
               "Por peligrosidad")

    _agregar_nota(fig,
        "H2: Los delitos se concentran en zonas geográficas específicas.\n"
        "Grupos separados por ciudad confirman patrones espaciales diferenciados.")

    plt.tight_layout()
    _guardar(fig, "pca_h2_concentracion_geografica.png", "H2 concentración geográfica")


def plot_h3_pc5_pc2(df):
    """H3 — Tipo de delito por ciudad y tiempo: PC5 × PC2."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("H3 — Tipo de delito por ciudad y tiempo: PC5 × PC2",
                 fontsize=14, fontweight="bold", y=1.01)

    _scatter_ciudad(axes[0], df, "pc5", "pc2")
    _estilo_ax(axes[0], "PC5 (peligrosidad / tipo agresión)", "PC2 (hora del delito)",
               "Por ciudad")

    _scatter_peligrosidad(axes[1], df, "pc5", "pc2")
    _estilo_ax(axes[1], "PC5 (peligrosidad / tipo agresión)", "PC2 (hora del delito)",
               "Por peligrosidad")

    _agregar_nota(fig,
        "H3: El tipo de delito (peligrosidad/agresión) varía según la hora y la ciudad.\n"
        "Separación en PC5 indica diferencias en el perfil de agresividad por momento del día.")

    plt.tight_layout()
    _guardar(fig, "pca_h3_tipo_delito_ciudad_tiempo.png", "H3 tipo de delito por ciudad y tiempo")


def plot_pc1_pc5(df):
    """PC1 × PC5 — Zonas geográficas vs peligrosidad."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("PC1 × PC5 — Zonas geográficas vs peligrosidad",
                 fontsize=14, fontweight="bold", y=1.01)

    _scatter_ciudad(axes[0], df, "pc1", "pc5")
    _estilo_ax(axes[0], "PC1 (posición geográfica)", "PC5 (peligrosidad / tipo agresión)",
               "Por ciudad")

    _scatter_peligrosidad(axes[1], df, "pc1", "pc5")
    _estilo_ax(axes[1], "PC1 (posición geográfica)", "PC5 (peligrosidad / tipo agresión)",
               "Por peligrosidad")

    _agregar_nota(fig,
        "PC1 × PC5: Relaciona la ubicación geográfica con el nivel de peligrosidad del delito.\n"
        "Permite identificar zonas con mayor concentración de delitos violentos.")

    plt.tight_layout()
    _guardar(fig, "pca_pc1_pc5_zonas_peligrosidad.png", "zonas geográficas vs peligrosidad")


def _agregar_nota(fig, texto):
    """Agrega nota de interpretación al pie de la figura."""
    fig.text(0.5, -0.03, texto, ha="center", va="top",
             fontsize=8.5, color="#555555", style="italic",
             wrap=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = cargar_datos()

    print("\n Generando gráficas PCA ...")
    plot_pc1_pc2(df)
    plot_h1_pc2_pc3(df)
    plot_h2_pc1_pc4(df)
    plot_h3_pc5_pc2(df)
    plot_pc1_pc5(df)

    print(f"\n{'='*50}")
    print(f"  ✔ 5 gráficas guardadas en: {RUTA_OUTPUT}/")
    print(f"  pca_pc1_pc2_vision_general.png")
    print(f"  pca_h1_patrones_temporales.png")
    print(f"  pca_h2_concentracion_geografica.png")
    print(f"  pca_h3_tipo_delito_ciudad_tiempo.png")
    print(f"  pca_pc1_pc5_zonas_peligrosidad.png")
    print(f"{'='*50}")