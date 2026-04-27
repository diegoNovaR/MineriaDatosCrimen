"""
visualizer.py
Visualizaciones del proyecto de crimen.

Gráfica 1: Heatmap de crímenes por mes y año (las 3 ciudades juntas, subplots)
Gráfica 3: Mapa de calor animado por mes con slider temporal (folium HeatMapWithTime)

Uso:
    python src/visualizer.py --plot heatmap_tiempo
    python src/visualizer.py --plot mapa_animado
    python src/visualizer.py --plot ambos
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import folium
from folium.plugins import HeatMapWithTime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_FILES

# ─── Configuración de salida ──────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "visualizaciones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

CITY_LABELS = {
    "chicago":       "Chicago",
    "philadelphia":  "Philadelphia",
    "san_francisco": "San Francisco",
}

# Paleta de colores por ciudad para consistencia visual
CITY_COLORS = {
    "chicago":       "Blues",
    "philadelphia":  "Oranges",
    "san_francisco": "Greens",
}


# ─── Carga de datos ───────────────────────────────────────────────────────────

def _load_all() -> dict[str, pd.DataFrame]:
    """Carga los 3 CSVs procesados y retorna {ciudad: df}."""
    data = {}
    for city, path in PROCESSED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"CSV procesado no encontrado: {path}\n"
                "Ejecuta primero: python main.py"
            )
        df = pd.read_csv(path, low_memory=False, parse_dates=["date"])
        # Asegurar tipos correctos
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
        data[city] = df
        print(f"  ✓ {CITY_LABELS[city]:<15}: {len(df):>10,} filas")
    return data


# ─── GRÁFICA 1: Heatmap mes vs año ───────────────────────────────────────────

def plot_heatmap_tiempo(data: dict[str, pd.DataFrame]) -> None:
    """
    Heatmap de crímenes por mes y año para las 3 ciudades.
    Genera un subplot por ciudad, todos en la misma figura.
    Guarda la imagen en data/visualizaciones/heatmap_mes_anio.png
    """
    print("\n── Generando Heatmap mes vs año ──")

    cities   = list(data.keys())
    n_cities = len(cities)

    fig, axes = plt.subplots(
        1, n_cities,
        figsize=(7 * n_cities, 6),
        constrained_layout=True
    )
    fig.suptitle(
        "Crímenes por Mes y Año — Chicago | Philadelphia | San Francisco",
        fontsize=15, fontweight="bold", y=1.02
    )

    for ax, city in zip(axes, cities):
        df = data[city].dropna(subset=["year", "month"])

        # Tabla pivote: filas=año, columnas=mes, valores=conteo
        pivot = (
            df.groupby(["year", "month"], observed=True)
              .size()
              .reset_index(name="total")
              .pivot(index="year", columns="month", values="total")
              .fillna(0)
              .astype(int)
        )
        # Renombrar columnas con nombres de mes
        pivot.columns = [MESES[m] for m in pivot.columns]
        pivot.index   = pivot.index.astype(int)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=CITY_COLORS[city],
            annot=True,
            fmt=",d",
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Nº de crímenes", "shrink": 0.8},
            annot_kws={"size": 7},
        )

        ax.set_title(CITY_LABELS[city], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Mes",  fontsize=10)
        ax.set_ylabel("Año",  fontsize=10)
        ax.tick_params(axis="x", rotation=0,  labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    out_path = OUTPUT_DIR / "heatmap_mes_anio.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Guardado en: {out_path}")


# ─── GRÁFICA 3: Mapa animado por mes (HeatMapWithTime) ───────────────────────

# Configuración por ciudad: centro del mapa y zoom óptimo para ver barrios
CITY_MAP_CONFIG = {
    "chicago": {
        "center": (41.8781, -87.6298),
        "zoom":   11,
        "label":  "Chicago",
    },
    "philadelphia": {
        "center": (39.9526, -75.1652),
        "zoom":   12,
        "label":  "Philadelphia",
    },
    "san_francisco": {
        "center": (37.7749, -122.4194),
        "zoom":   12,
        "label":  "San Francisco",
    },
}


def _build_mapa_ciudad(
    city: str,
    df: pd.DataFrame,
    sample_mes: int = 5_000,
) -> None:
    """
    Genera el mapa animado por mes para UNA ciudad.
    Guarda en: data/visualizaciones/mapa_animado_{city}.html
    """
    cfg = CITY_MAP_CONFIG[city]
    print(f"\n  [{cfg['label']}] Construyendo frames...")

    df = df.dropna(subset=["latitude", "longitude", "year", "month"]).copy()
    df["year"]    = df["year"].astype(int)
    df["month"]   = df["month"].astype(int)
    df["periodo"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    periodos      = sorted(df["periodo"].unique())

    heat_data = []
    for periodo in periodos:
        grupo   = df[df["periodo"] == periodo]
        muestra = grupo.sample(min(len(grupo), sample_mes), random_state=42)
        puntos  = [[row.latitude, row.longitude, 1] for row in muestra.itertuples()]
        heat_data.append(puntos)

    print(f"  [{cfg['label']}] Períodos: {len(periodos)}  ({periodos[0]} → {periodos[-1]})")

    mapa = folium.Map(
        location=cfg["center"],
        zoom_start=cfg["zoom"],
        tiles="CartoDB dark_matter",
    )

    HeatMapWithTime(
        data=heat_data,
        index=periodos,
        radius=8,               # radio pequeño → se ve la variación por barrio
        blur=0.7,
        min_opacity=0.3,
        max_opacity=0.85,
        scale_radius=False,     # radio fijo sin importar el zoom
        use_local_extrema=True, # normaliza por mes → resalta cambios relativos
        auto_play=False,
        display_index=True,
        min_speed=0.5,
        max_speed=5,
        speed_step=0.5,
    ).add_to(mapa)

    # Marcador del centro de la ciudad
    folium.Marker(
        location=cfg["center"],
        tooltip=cfg["label"],
        icon=folium.Icon(color="white", icon="info-sign"),
    ).add_to(mapa)

    out_path = OUTPUT_DIR / f"mapa_animado_{city}.html"
    mapa.save(str(out_path))
    print(f"  [{cfg['label']}] ✓ Guardado en: {out_path}")


def plot_mapa_animado(data: dict[str, pd.DataFrame], sample_per_city: int = 5_000) -> None:
    """
    Genera un mapa animado por mes por cada ciudad (3 archivos HTML).
    Cada HTML tiene zoom óptimo para ver la expansión dentro de la ciudad.

    Parámetro sample_per_city: máx. puntos por mes por ciudad.

    Guarda en: data/visualizaciones/mapa_animado_{ciudad}.html
    """
    print("\n── Generando Mapas animados por mes (uno por ciudad) ──")
    for city, df in data.items():
        _build_mapa_ciudad(city, df, sample_mes=sample_per_city)
    print("\n  → Abre cada HTML en el navegador y usa el slider para ver la expansión mensual")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualizaciones del proyecto de crimen")
    parser.add_argument(
        "--plot",
        choices=["heatmap_tiempo", "mapa_animado", "ambos"],
        default="ambos",
        help="Qué gráfica generar (default: ambos)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=8_000,
        help="Máx. puntos por ciudad por mes en el mapa animado (default: 8000)",
    )
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  Cargando datos procesados...")
    print("="*55)
    data = _load_all()

    if args.plot in ("heatmap_tiempo", "ambos"):
        plot_heatmap_tiempo(data)

    if args.plot in ("mapa_animado", "ambos"):
        plot_mapa_animado(data, sample_per_city=args.sample)

    print("\n" + "="*55)
    print("  Visualizaciones completadas ✓")
    print(f"  Archivos en: {OUTPUT_DIR}")
    print("="*55)


if __name__ == "__main__":
    main()