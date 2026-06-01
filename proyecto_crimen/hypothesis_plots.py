"""
hypothesis_plots.py
-------------------
Genera 5 scatter plots PCA para análisis de hipótesis de crimen urbano.
Salida: outputs/pca/  (PNG, dpi=150)

Plots generados:
  1. pc1_vs_pc2_general.png        — Visión general: geografía vs tiempo
  2. pc2_vs_pc3_h1_temporal.png    — H1: Patrones temporales (hora vs mes)
  3. pc1_vs_pc4_h2_geografico.png  — H2: Concentración geográfica
  4. pc5_vs_pc2_h3_tipo_ciudad.png — H3: Tipos de delito por ciudad y tiempo
  5. pc1_vs_pc5_peligrosidad.png   — Zonas geográficas vs peligrosidad

NOTAS SOBRE ALINEACIÓN:
  - crime_pca.csv: N filas × 16 PCs (pc1..pc16), mismo orden que crime_features.csv
  - crime_unified.csv: N filas con metadatos originales (ciudad, es_peligroso, etc.)
  - La alineación es por índice de fila (el PCA no hace shuffle de filas).
  - Se valida que los conteos coincidan antes de hacer el merge.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ─── Rutas ────────────────────────────────────────────────────────────────────
RUTA_PCA      = os.path.join("data", "processed", "crime_pca.csv")
RUTA_UNIFIED  = os.path.join("data", "processed", "crime_unified.csv")
RUTA_OUTPUTS  = os.path.join("outputs", "pca")

# ─── Paletas ──────────────────────────────────────────────────────────────────
PALETA_CIUDAD = {
    "chicago":       "#2196F3",
    "philadelphia":  "#FF5722",
    "san_francisco": "#4CAF50",
}

PALETA_PELIGROSO = {
    True:  "#E53935",
    False: "#90A4AE",
}

# Orden cronológico natural del día
PALETA_PERIODO = {
    "madrugada": "#3F51B5",   # azul oscuro
    "mañana":    "#FFC107",   # ámbar
    "tarde":     "#FF7043",   # naranja
    "noche":     "#212121",   # casi negro
}

ALPHA      = 0.30
POINT_SIZE = 6
DPI        = 150


# ─── Carga y enriquecimiento ──────────────────────────────────────────────────

def cargar_datos() -> pd.DataFrame:
    """
    Carga crime_pca.csv y le añade 'ciudad' y 'es_peligroso' desde
    crime_unified.csv mediante alineación por índice de fila.

    Premisa válida: ambos archivos tienen el mismo número de filas
    en el mismo orden (el pipeline de PCA no hace shuffle).

    Si crime_pca.csv ya contiene esas columnas se usan directamente.
    """
    if not os.path.exists(RUTA_PCA):
        raise FileNotFoundError(f"No se encontró: {RUTA_PCA}")

    df_pca = pd.read_csv(RUTA_PCA, low_memory=False)
    print(f"  [carga] crime_pca.csv  → {len(df_pca):,} filas, {df_pca.shape[1]} cols")

    # Verificar que las columnas PC existen
    pc_cols = [c for c in df_pca.columns if c.startswith("pc")]
    if len(pc_cols) < 5:
        raise ValueError(
            f"Se esperaban al menos 5 columnas 'pc*' en crime_pca.csv, "
            f"se encontraron: {pc_cols}"
        )

    necesita = {"ciudad", "es_peligroso", "periodo_dia"} - set(df_pca.columns)

    if necesita:
        # ── Cargar unified ──────────────────────────────────────────────────
        if not os.path.exists(RUTA_UNIFIED):
            raise FileNotFoundError(
                f"Faltan columnas {necesita} en crime_pca.csv "
                f"y no se encontró {RUTA_UNIFIED}"
            )

        df_uni = pd.read_csv(
            RUTA_UNIFIED,
            low_memory=False,
            usecols=["ciudad", "es_peligroso", "periodo_dia"],
        )
        print(f"  [carga] crime_unified.csv → {len(df_uni):,} filas")

        # ── Validar alineación ──────────────────────────────────────────────
        if len(df_uni) != len(df_pca):
            raise ValueError(
                f"Número de filas no coincide:\n"
                f"  crime_pca.csv     : {len(df_pca):,}\n"
                f"  crime_unified.csv : {len(df_uni):,}\n"
                f"El pipeline de PCA debe preservar el orden y cantidad de filas."
            )

        # ── Merge por posición (reset_index garantiza 0-based en ambos) ────
        df_pca = df_pca.reset_index(drop=True)
        df_uni = df_uni.reset_index(drop=True)

        for col in necesita:
            df_pca[col] = df_uni[col].values
        print(f"  [enrich] Columnas añadidas desde crime_unified.csv: {necesita}")

    else:
        print("  [info] 'ciudad' y 'es_peligroso' ya presentes en crime_pca.csv")

    # ── Normalizar tipos ────────────────────────────────────────────────────
    df_pca["ciudad"]       = df_pca["ciudad"].str.lower().str.strip()
    df_pca["es_peligroso"] = df_pca["es_peligroso"].astype(bool)
    df_pca["periodo_dia"]  = df_pca["periodo_dia"].str.lower().str.strip()

    # ── Diagnóstico ─────────────────────────────────────────────────────────
    print(f"  [info] Ciudades    : {df_pca['ciudad'].value_counts().to_dict()}")
    print(f"  [info] Peligrosos  : {df_pca['es_peligroso'].sum():,} / {len(df_pca):,}")
    print(f"  [info] Componentes : {[c for c in df_pca.columns if c.startswith('pc')]}")

    return df_pca


# ─── Helpers de graficado ─────────────────────────────────────────────────────

def _scatter_ciudad(ax, df, xcol, ycol, alpha=ALPHA, s=POINT_SIZE):
    """Scatter coloreado por ciudad."""
    for ciudad, color in PALETA_CIUDAD.items():
        mask = df["ciudad"] == ciudad
        if mask.any():
            ax.scatter(
                df.loc[mask, xcol], df.loc[mask, ycol],
                c=color, alpha=alpha, s=s, linewidths=0,
                label=ciudad.replace("_", " ").title(),
            )


def _scatter_peligrosidad(ax, df, xcol, ycol, alpha=ALPHA, s=POINT_SIZE):
    """Scatter coloreado por es_peligroso."""
    etiquetas = {False: "No peligroso", True: "Peligroso"}
    for valor, color in PALETA_PELIGROSO.items():
        mask = df["es_peligroso"] == valor
        if mask.any():
            ax.scatter(
                df.loc[mask, xcol], df.loc[mask, ycol],
                c=color, alpha=alpha, s=s, linewidths=0,
                label=etiquetas[valor],
            )


def _scatter_periodo(ax, df, xcol, ycol, alpha=ALPHA, s=POINT_SIZE):
    """Scatter coloreado por periodo_dia (orden cronológico natural)."""
    for periodo, color in PALETA_PERIODO.items():
        mask = df["periodo_dia"] == periodo
        if mask.any():
            ax.scatter(
                df.loc[mask, xcol], df.loc[mask, ycol],
                c=color, alpha=alpha, s=s, linewidths=0,
                label=periodo.capitalize(),
            )


def _estilo_ax(ax, xlabel, ylabel, titulo):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(titulo, fontsize=11, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(
        loc="upper right", fontsize=8, markerscale=2.0,
        framealpha=0.7, edgecolor="#cccccc",
    )


def _guardar(fig, nombre):
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [guardado] {ruta}")


# ─── Plots individuales ───────────────────────────────────────────────────────

def plot_general_pc1_pc2(df: pd.DataFrame):
    """PC1 vs PC2 — visión general: geografía vs tiempo. Color por ciudad."""
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_ciudad(ax, df, "pc1", "pc2")
    _estilo_ax(
        ax,
        xlabel="PC1  (componente geográfica principal)",
        ylabel="PC2  (componente temporal — hora)",
        titulo="PC1 × PC2 — Visión general: Geografía vs. Tiempo\n(coloreado por ciudad)",
    )
    fig.tight_layout()
    _guardar(fig, "pc1_vs_pc2_general.png")


def plot_h1_temporal(df: pd.DataFrame):
    """H1 — PC2 vs PC3: patrones temporales. Color por periodo_dia."""
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_periodo(ax, df, "pc2", "pc3")
    _estilo_ax(
        ax,
        xlabel="PC2  (componente temporal — hora del día)",
        ylabel="PC3  (componente temporal — mes del año)",
        titulo="H1 — PC2 × PC3: Patrones Temporales\n"
               "Hora del día vs. Mes del año  (coloreado por periodo del día)",
    )
    fig.tight_layout()
    _guardar(fig, "pc2_vs_pc3_h1_temporal.png")


def plot_h2_geografico(df: pd.DataFrame):
    """H2 — PC1 vs PC4: concentración geográfica. Color por ciudad."""
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_ciudad(ax, df, "pc1", "pc4")
    _estilo_ax(
        ax,
        xlabel="PC1  (latitud/longitud — componente geográfica 1)",
        ylabel="PC4  (latitud/longitud + ciudad — componente geográfica 2)",
        titulo="H2 — PC1 × PC4: Concentración Geográfica\n(coloreado por ciudad)",
    )
    fig.tight_layout()
    _guardar(fig, "pc1_vs_pc4_h2_geografico.png")


def plot_h3_tipo_ciudad(df: pd.DataFrame):
    """H3 — PC5 vs PC2: tipos de delito por ciudad y tiempo. Doble panel."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "H3 — PC5 × PC2: Tipo de Delito por Ciudad y Franja Horaria",
        fontsize=12, fontweight="bold",
    )

    _scatter_ciudad(axes[0], df, "pc5", "pc2")
    _estilo_ax(
        axes[0],
        xlabel="PC5  (peligrosidad / agresión)",
        ylabel="PC2  (hora del día)",
        titulo="Color por ciudad",
    )

    _scatter_peligrosidad(axes[1], df, "pc5", "pc2")
    _estilo_ax(
        axes[1],
        xlabel="PC5  (peligrosidad / agresión)",
        ylabel="PC2  (hora del día)",
        titulo="Color por peligrosidad",
    )

    fig.tight_layout()
    _guardar(fig, "pc5_vs_pc2_h3_tipo_ciudad.png")


def plot_peligrosidad_geografica(df: pd.DataFrame):
    """PC1 vs PC5: zonas geográficas vs peligrosidad. Solo color=peligrosidad."""
    fig, ax = plt.subplots(figsize=(9, 6))

    _scatter_peligrosidad(ax, df, "pc1", "pc5")

    _estilo_ax(
        ax,
        xlabel="PC1  (componente geográfica principal — latitud/longitud)",
        ylabel="PC5  (peligrosidad / agresión)",
        titulo="PC1 × PC5 — Zonas Geográficas vs. Peligrosidad\n"
               "(coloreado por peligrosidad)",
    )
    fig.tight_layout()
    _guardar(fig, "pc1_vs_pc5_peligrosidad.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def generar_todos():
    print(f"\n{'='*55}")
    print(f"  GENERANDO PLOTS DE HIPÓTESIS PCA")
    print(f"{'='*55}")

    df = cargar_datos()

    print("\n  → Plot 1/5: PC1 × PC2 (visión general)")
    plot_general_pc1_pc2(df)

    print("  → Plot 2/5: H1 — PC2 × PC3 (patrones temporales)")
    plot_h1_temporal(df)

    print("  → Plot 3/5: H2 — PC1 × PC4 (concentración geográfica)")
    plot_h2_geografico(df)

    print("  → Plot 4/5: H3 — PC5 × PC2 (tipo de delito por ciudad y tiempo)")
    plot_h3_tipo_ciudad(df)

    print("  → Plot 5/5: PC1 × PC5 (zonas geográficas vs peligrosidad)")
    plot_peligrosidad_geografica(df)

    print(f"\n  ✔ 5 gráficas guardadas en: {RUTA_OUTPUTS}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    generar_todos()