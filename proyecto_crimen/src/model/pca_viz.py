import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.translations import CIUDADES_LABEL, CATEGORIA_LABEL

RUTA_OUTPUTS = os.path.join("outputs", "pca")

# ─── Controla cuántos puntos se muestran en las visualizaciones ──────────────
# Se distribuye equitativamente entre las 3 ciudades (33% cada una)
N_MUESTRA = 30_000

MARKER_SIZE    = 2.5
MARKER_OPACITY = 0.4

COLORES_CIUDAD = {
    "Chicago":       "#2196F3",
    "Philadelphia":  "#F44336",
    "San Francisco": "#4CAF50",
}
COLORES_PELIGROSO = {
    "Peligroso":     "#EF5350",
    "No peligroso":  "#42A5F5",
}


def _muestra_estratificada(d: pd.DataFrame) -> pd.DataFrame:
    """Toma N_MUESTRA filas distribuidas equitativamente entre ciudades."""
    if "ciudad_label" not in d.columns or len(d) <= N_MUESTRA:
        return d.reset_index(drop=True)

    ciudades = d["ciudad_label"].unique()
    n_por_ciudad = N_MUESTRA // len(ciudades)
    frames = []
    rng = np.random.default_rng(42)

    for ciudad in ciudades:
        sub = d[d["ciudad_label"] == ciudad]
        n   = min(n_por_ciudad, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        frames.append(sub.iloc[idx])

    resultado = pd.concat(frames, ignore_index=True)
    print(f"  [muestra] {len(resultado):,} puntos "
          f"({n_por_ciudad:,} por ciudad aprox)")
    return resultado


def graficar_interactivo(df_pca_2d: pd.DataFrame,
                         df_pca_3d: pd.DataFrame,
                         df_unificado: pd.DataFrame) -> None:
    """
    Genera visualizaciones interactivas Plotly.
    Alinea PCA con el unificado por índice de fila para el hover.
    """
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  PCA — VISUALIZACIÓN INTERACTIVA (Plotly)")
    print(f"{'='*55}")

    # Alinear por índice de fila
    n = min(len(df_pca_2d), len(df_pca_3d), len(df_unificado))
    pca2 = df_pca_2d.iloc[:n].reset_index(drop=True)
    pca3 = df_pca_3d.iloc[:n].reset_index(drop=True)
    unif = df_unificado.iloc[:n].reset_index(drop=True)

    # Combinar PCA con datos reales
    d2 = _combinar(pca2, unif)
    d3 = _combinar(pca3, unif)

    # Muestra estratificada por ciudad
    d2m = _muestra_estratificada(d2)
    d3m = _muestra_estratificada(d3)

    # Generar gráficas
    _scatter_2d(d2m, "ciudad_label", COLORES_CIUDAD,
                "PCA 2D — Distribución por ciudad",
                "Ciudad", "pca_2d_ciudad.html")

    _scatter_2d(d2m, "peligroso_label", COLORES_PELIGROSO,
                "PCA 2D — Peligroso vs No peligroso",
                "Peligrosidad", "pca_2d_peligroso.html")

    _scatter_3d(d3m, "ciudad_label", COLORES_CIUDAD,
                "PCA 3D — Distribución por ciudad",
                "Ciudad", "pca_3d_ciudad.html")

    _scatter_3d(d3m, "peligroso_label", COLORES_PELIGROSO,
                "PCA 3D — Peligroso vs No peligroso",
                "Peligrosidad", "pca_3d_peligroso.html")

    print(f"\n  ✔ Visualizaciones guardadas en: {RUTA_OUTPUTS}/")


# ─── Combinar PCA con datos reales ───────────────────────────────────────────

def _combinar(df_pca: pd.DataFrame, df_unif: pd.DataFrame) -> pd.DataFrame:
    """Une componentes PCA con datos reales del unificado por índice."""
    d = df_pca.copy()
    d["ciudad_label"] = df_unif["ciudad"].map(CIUDADES_LABEL).fillna(df_unif["ciudad"])
    d["categoria_label"] = df_unif["categoria_delito"].map(CATEGORIA_LABEL).fillna(df_unif["categoria_delito"])
    d["peligroso_label"] = df_unif["es_peligroso"].astype(bool).map(
        {True: "Peligroso", False: "No peligroso"})
    d["hora"]     = df_unif["hora"].values
    d["mes"]      = df_unif["mes"].values
    d["latitud"]  = df_unif["latitud"].values
    d["longitud"] = df_unif["longitud"].values

    if "fecha" in df_unif.columns:
        d["fecha"] = pd.to_datetime(df_unif["fecha"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    return d


# ─── Scatter 2D ───────────────────────────────────────────────────────────────

def _scatter_2d(d: pd.DataFrame, color_col: str, color_map: dict,
                titulo: str, legend_title: str, nombre: str) -> None:
    custom = _custom_data(d)
    hover  = _hovertemplate()

    fig = px.scatter(
        d, x="pc1", y="pc2",
        color=color_col,
        color_discrete_map=color_map,
        custom_data=custom,
        title=f"{titulo}<br><sup>Hover sobre un punto para ver datos reales del delito</sup>",
        labels={"pc1": "Componente Principal 1",
                "pc2": "Componente Principal 2",
                color_col: legend_title},
        opacity=MARKER_OPACITY,
    )
    fig.update_traces(marker=dict(size=MARKER_SIZE), hovertemplate=hover)
    fig.update_layout(
        template="plotly_white",
        title_font_size=14,
        margin=dict(t=80, b=40, l=40, r=40),
    )
    fig.update_xaxes(zeroline=True, zerolinecolor="lightgray")
    fig.update_yaxes(zeroline=True, zerolinecolor="lightgray")
    _guardar(fig, nombre)


# ─── Scatter 3D ───────────────────────────────────────────────────────────────

def _scatter_3d(d: pd.DataFrame, color_col: str, color_map: dict,
                titulo: str, legend_title: str, nombre: str) -> None:
    if "pc3" not in d.columns:
        print(f"  [AVISO] pc3 no disponible. Omitiendo {nombre}.")
        return

    custom = _custom_data(d)
    hover  = _hovertemplate()

    fig = px.scatter_3d(
        d, x="pc1", y="pc2", z="pc3",
        color=color_col,
        color_discrete_map=color_map,
        custom_data=custom,
        title=f"{titulo}<br><sup>Hover sobre un punto para ver datos reales del delito</sup>",
        labels={"pc1": "PC1", "pc2": "PC2", "pc3": "PC3",
                color_col: legend_title},
        opacity=MARKER_OPACITY,
    )
    fig.update_traces(marker=dict(size=max(1.5, MARKER_SIZE - 1)), hovertemplate=hover)
    fig.update_layout(
        template="plotly_white",
        title_font_size=14,
        margin=dict(t=80, b=0, l=0, r=0),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    )
    _guardar(fig, nombre)


# ─── Hover con datos reales ──────────────────────────────────────────────────

def _custom_data(d: pd.DataFrame) -> list:
    """Columnas del unificado que van al hover, en orden fijo."""
    return ["ciudad_label", "categoria_label", "peligroso_label",
            "hora", "mes", "fecha", "latitud", "longitud"]


def _hovertemplate() -> str:
    """Template del hover mostrando datos reales del delito."""
    return (
        "<b>Ciudad</b>: %{customdata[0]}<br>"
        "<b>Categoría</b>: %{customdata[1]}<br>"
        "<b>Peligrosidad</b>: %{customdata[2]}<br>"
        "<b>Hora</b>: %{customdata[3]}<br>"
        "<b>Mes</b>: %{customdata[4]}<br>"
        "<b>Fecha</b>: %{customdata[5]}<br>"
        "<b>Latitud</b>: %{customdata[6]}<br>"
        "<b>Longitud</b>: %{customdata[7]}<br>"
        "<extra></extra>"
    )


# ─── Helper ──────────────────────────────────────────────────────────────────

def _guardar(fig: go.Figure, nombre: str) -> None:
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.write_html(ruta, include_plotlyjs="cdn")
    print(f"  [html] Guardado → {ruta}")