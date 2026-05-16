import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.translations import CIUDADES_LABEL, TIPO_DETALLE_LABEL

RUTA_OUTPUTS = os.path.join("outputs")


def analizar_granularidad(datasets: dict) -> None:
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)

    for ciudad, df in datasets.items():
        print(f"\n{'='*55}")
        print(f"  GRANULARIDAD: {ciudad.replace('_', ' ').upper()}")
        print(f"{'='*55}")

        _consola_tipo_detalle(ciudad, df)
        _consola_descripcion(ciudad, df)

    _grafica_top_tipo_detalle(datasets)
    _grafica_top_descripcion(datasets)

    print(f"\n  ✔ Análisis de granularidad completado.")


# ─── Consola ──────────────────────────────────────────────────────────────────

def _consola_tipo_detalle(ciudad: str, df: pd.DataFrame) -> None:
    col = "tipo_delito_detalle"
    if col not in df.columns:
        print(f"  [AVISO] No existe '{col}'.")
        return

    conteo = df[col].value_counts()
    total  = len(df)

    print(f"\n  [TIPO_DELITO_DETALLE] — {conteo.nunique()} valores únicos")
    print(f"  {'Tipo':<45} {'Cant':>8}  {'%':>6}")
    print(f"  {'-'*63}")
    for tipo, n in conteo.items():
        print(f"  {str(tipo):<45} {n:>8,}  {n/total*100:>5.1f}%")


def _consola_descripcion(ciudad: str, df: pd.DataFrame) -> None:
    col = "descripcion"
    if col not in df.columns:
        print(f"\n  [DESCRIPCION] — no disponible para {ciudad}")
        return

    conteo = df[col].value_counts()
    total  = len(df)

    print(f"\n  [DESCRIPCION] — {conteo.nunique()} valores únicos  (top 15)")
    print(f"  {'Descripción':<55} {'Cant':>8}  {'%':>6}")
    print(f"  {'-'*73}")
    for desc, n in conteo.head(15).items():
        print(f"  {str(desc):<55} {n:>8,}  {n/total*100:>5.1f}%")


# ─── Gráficas ─────────────────────────────────────────────────────────────────

def _grafica_top_tipo_detalle(datasets: dict) -> None:
    """Top 15 tipo_delito_detalle por ciudad — barras horizontales."""
    col = "tipo_delito_detalle"
    ciudades = [c for c, df in datasets.items() if col in df.columns]
    if not ciudades:
        return

    n = len(ciudades)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 7))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Granularidad — Top 15 tipos de delito por ciudad (nivel medio)",
        fontsize=13, fontweight="bold", y=1.01
    )

    COLORES = ["#2196F3", "#F44336", "#4CAF50"]

    for ax, ciudad, color in zip(axes, ciudades, COLORES):
        df    = datasets[ciudad]
        top   = df[col].value_counts().head(15)
        # Traducir etiquetas al español
        etiquetas = [TIPO_DETALLE_LABEL.get(t, t) for t in top.index[::-1]]
        bars  = ax.barh(etiquetas, top.values[::-1],
                        color=color, edgecolor="white")
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=12)
        ax.set_xlabel("Número de delitos")
        ax.set_ylabel("Tipo de delito")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.bar_label(bars, fmt=lambda x: f"{int(x):,}", padding=3, fontsize=8)

    plt.tight_layout()
    _guardar(fig, "granularidad_tipo_detalle.png")


def _grafica_top_descripcion(datasets: dict) -> None:
    """Top 15 descripcion por ciudad — solo donde existe."""
    col = "descripcion"
    ciudades = [c for c, df in datasets.items() if col in df.columns]
    if not ciudades:
        print(f"  [AVISO] Ningún dataset tiene columna '{col}'. Omitiendo gráfica.")
        return

    n = len(ciudades)
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 7))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Granularidad — Top 15 descripciones de delito (nivel fino)",
        fontsize=13, fontweight="bold", y=1.01
    )

    COLORES = ["#2196F3", "#F44336", "#4CAF50"]

    for ax, ciudad, color in zip(axes, ciudades, COLORES):
        df   = datasets[ciudad]
        top  = df[col].value_counts().head(15)
        bars = ax.barh(top.index[::-1], top.values[::-1],
                       color=color, edgecolor="white")
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=12)
        ax.set_xlabel("Número de delitos")
        ax.set_ylabel("Descripción")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.bar_label(bars, fmt=lambda x: f"{int(x):,}", padding=3, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    _guardar(fig, "granularidad_descripcion.png")


def _guardar(fig: plt.Figure, nombre: str) -> None:
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")