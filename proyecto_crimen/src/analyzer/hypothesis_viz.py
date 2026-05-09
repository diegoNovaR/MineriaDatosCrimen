import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

RUTA_OUTPUTS = os.path.join("outputs")

ORDEN_PERIODO = ["madrugada", "mañana", "tarde", "noche"]
ORDEN_DIAS    = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DIAS_ES       = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
CIUDADES_LABEL = {
    "chicago":       "Chicago",
    "philadelphia":  "Philadelphia",
    "san_francisco": "San Francisco",
}
COLORES_CIUDAD = {
    "chicago":       "#2196F3",
    "philadelphia":  "#F44336",
    "san_francisco": "#4CAF50",
}


def generar_todas(datasets: dict) -> None:
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)
    grafica_1_periodo_dia(datasets)
    grafica_2_dia_semana(datasets)
    grafica_4_heatmap_geo(datasets)
    grafica_5_top_distritos(datasets)
    grafica_6_categoria_ciudad(datasets)
    grafica_7_categoria_vs_periodo(datasets)
    print(f"\n  ✔ Todas las gráficas guardadas en: {RUTA_OUTPUTS}/")


# ─── Gráfica 1: Delitos por periodo_dia (H1) ─────────────────────────────────

def grafica_1_periodo_dia(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(
        "H1 — Frecuencia de delitos por franja horaria",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        periodos = [p for p in ORDEN_PERIODO if p in df["periodo_dia"].values]
        conteo = (
            df["periodo_dia"]
            .value_counts()
            .reindex(periodos, fill_value=0)
        )
        bars = ax.bar(conteo.index, conteo.values,
                      color=COLORES_CIUDAD.get(ciudad, "#888"), edgecolor="white")
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=12)
        ax.set_xlabel("Franja horaria")
        ax.set_ylabel("Número de delitos")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.bar_label(bars, fmt=lambda x: f"{int(x):,}", padding=3, fontsize=8)
        ax.set_ylim(0, conteo.max() * 1.15)

    plt.tight_layout()
    _guardar(fig, "grafica_1_periodo_dia.png")


# ─── Gráfica 2: Delitos por dia_semana (H1) ──────────────────────────────────

def grafica_2_dia_semana(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle(
        "H1 — Frecuencia de delitos por día de la semana",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        conteo = (
            df["dia_semana"]
            .str.lower()
            .value_counts()
            .reindex(ORDEN_DIAS, fill_value=0)
        )
        colores = [
            "#F44336" if d in ["saturday", "sunday"] else COLORES_CIUDAD.get(ciudad, "#888")
            for d in ORDEN_DIAS
        ]
        bars = ax.bar(DIAS_ES, conteo.values, color=colores, edgecolor="white")
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=12)
        ax.set_xlabel("Día de la semana")
        ax.set_ylabel("Número de delitos")
        ax.set_xticklabels(DIAS_ES, rotation=35, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_ylim(0, conteo.max() * 1.15)
        ax.bar_label(bars, fmt=lambda x: f"{int(x):,}", padding=3, fontsize=7)

    # Leyenda fin de semana
    from matplotlib.patches import Patch
    legend = [Patch(color="#F44336", label="Fin de semana")]
    fig.legend(handles=legend, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    _guardar(fig, "grafica_2_dia_semana.png")


# ─── Gráfica 4: Heatmap geográfico lat/lon (H2) ──────────────────────────────

def grafica_4_heatmap_geo(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "H2 — Concentración geográfica de delitos (densidad lat/lon)",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        df_geo = df[["latitud", "longitud"]].dropna()
        h = ax.hist2d(
            df_geo["longitud"], df_geo["latitud"],
            bins=80, cmap="YlOrRd",
        )
        fig.colorbar(h[3], ax=ax, label="Densidad de delitos")
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=12)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

    plt.tight_layout()
    _guardar(fig, "grafica_4_heatmap_geo.png")


# ─── Gráfica 5: Top 10 distritos (H2) ────────────────────────────────────────

def grafica_5_top_distritos(datasets: dict) -> None:
    ciudades_con_distrito = {
        c: df for c, df in datasets.items()
        if "distrito" in df.columns and df["distrito"].notna().sum() > 0
    }

    if not ciudades_con_distrito:
        print("  [AVISO] Ningún dataset tiene columna 'distrito' disponible. Omitiendo gráfica 5.")
        return

    n = len(ciudades_con_distrito)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "H2 — Top 10 distritos con mayor concentración de delitos",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, (ciudad, df) in zip(axes, ciudades_con_distrito.items()):
        top = (
            df["distrito"]
            .astype(str)
            .value_counts()
            .head(10)
        )
        bars = ax.barh(top.index[::-1], top.values[::-1],
                       color=COLORES_CIUDAD.get(ciudad, "#888"), edgecolor="white")
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=12)
        ax.set_xlabel("Número de delitos")
        ax.set_ylabel("Distrito")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.bar_label(bars, fmt=lambda x: f"{int(x):,}", padding=3, fontsize=8)

    plt.tight_layout()
    _guardar(fig, "grafica_5_top_distritos.png")


# ─── Gráfica 6: Categoria_delito por ciudad (H3) ─────────────────────────────

def grafica_6_categoria_ciudad(datasets: dict) -> None:
    # Construir tabla unificada
    frames = []
    for ciudad, df in datasets.items():
        if "categoria_delito" not in df.columns:
            continue
        conteo = df["categoria_delito"].value_counts().reset_index()
        conteo.columns = ["categoria_delito", "conteo"]
        conteo["ciudad"] = CIUDADES_LABEL.get(ciudad, ciudad)
        frames.append(conteo)

    if not frames:
        print("  [AVISO] No hay columna categoria_delito. Omitiendo gráfica 6.")
        return

    tabla = pd.concat(frames, ignore_index=True)
    categorias = tabla.groupby("categoria_delito")["conteo"].sum().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(13, 6))
    ciudades = list(CIUDADES_LABEL.values())
    x = np.arange(len(categorias))
    ancho = 0.25

    for i, ciudad_label in enumerate(ciudades):
        sub = tabla[tabla["ciudad"] == ciudad_label].set_index("categoria_delito")
        valores = [sub.loc[c, "conteo"] if c in sub.index else 0 for c in categorias]
        bars = ax.bar(x + i * ancho, valores, ancho,
                      label=ciudad_label,
                      color=list(COLORES_CIUDAD.values())[i],
                      edgecolor="white")

    ax.set_title("H3 — Categorías de delito por ciudad", fontsize=14, fontweight="bold")
    ax.set_xlabel("Categoría de delito")
    ax.set_ylabel("Número de delitos")
    ax.set_xticks(x + ancho)
    ax.set_xticklabels(categorias, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Ciudad")
    plt.tight_layout()
    _guardar(fig, "grafica_6_categoria_ciudad.png")


# ─── Gráfica 7: Categoria_delito vs periodo_dia heatmap (H3) ─────────────────

def grafica_7_categoria_vs_periodo(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "H3 — Tipo de delito según franja horaria por ciudad",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        if "categoria_delito" not in df.columns or "periodo_dia" not in df.columns:
            ax.set_visible(False)
            continue

        pivot = (
            df.groupby(["categoria_delito", "periodo_dia"])
            .size()
            .unstack(fill_value=0)
        )
        # Ordenar columnas por periodo
        cols_ord = [p for p in ORDEN_PERIODO if p in pivot.columns]
        pivot = pivot[cols_ord]

        # Normalizar por fila (% dentro de cada categoría)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        sns.heatmap(
            pivot_pct,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=0.4,
            ax=ax,
            cbar_kws={"label": "% del total por categoría"},
        )
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=12)
        ax.set_xlabel("Franja horaria")
        ax.set_ylabel("Categoría de delito")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    _guardar(fig, "grafica_7_categoria_vs_periodo.png")


# ─── Helper ───────────────────────────────────────────────────────────────────

def _guardar(fig: plt.Figure, nombre: str) -> None:
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")