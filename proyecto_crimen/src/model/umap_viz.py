import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.translations import CIUDADES_LABEL, CATEGORIA_LABEL

RUTA_OUTPUTS = os.path.join("outputs", "umap")

# ─── Controla cuántos puntos se muestran en las visualizaciones ──────────────
# Se distribuye equitativamente entre las 3 ciudades (33% cada una)
N_MUESTRA      = 30_000
MARKER_SIZE    = 2.5
MARKER_OPACITY = 0.4

COLORES_CIUDAD = {
    "Chicago":       "#2196F3",
    "Philadelphia":  "#F44336",
    "San Francisco": "#4CAF50",
}
COLORES_PELIGROSO = {
    "Peligroso":    "#EF5350",
    "No peligroso": "#42A5F5",
}


def _muestra_estratificada(d: pd.DataFrame) -> pd.DataFrame:
    """Toma N_MUESTRA filas distribuidas equitativamente entre ciudades."""
    if "ciudad_label" not in d.columns or len(d) <= N_MUESTRA:
        return d.reset_index(drop=True)

    ciudades     = d["ciudad_label"].unique()
    n_por_ciudad = N_MUESTRA // len(ciudades)
    frames = []
    rng = np.random.default_rng(42)

    for ciudad in ciudades:
        sub = d[d["ciudad_label"] == ciudad]
        n   = min(n_por_ciudad, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        frames.append(sub.iloc[idx])

    resultado = pd.concat(frames, ignore_index=True)
    print(f"  [muestra] {len(resultado):,} puntos ({n_por_ciudad:,} por ciudad aprox)")
    return resultado


def graficar_umap(df_umap: pd.DataFrame, df_unificado: pd.DataFrame) -> None:
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  UMAP — VISUALIZACIÓN INTERACTIVA (Plotly)")
    print(f"{'='*55}")

    n       = min(len(df_umap), len(df_unificado))
    umap_df = df_umap.iloc[:n].reset_index(drop=True)
    unif    = df_unificado.iloc[:n].reset_index(drop=True)

    d = _combinar(umap_df, unif)
    d = _muestra_estratificada(d)

    # Gráficas
    _scatter_neutral(d)
    _scatter_ciudad(d)
    _scatter_peligroso(d)
    _scatter_categoria(d)

    print(f"\n  ✔ Visualizaciones guardadas en: {RUTA_OUTPUTS}/")


# ─── Combinar UMAP con datos reales ──────────────────────────────────────────

def _combinar(df_umap: pd.DataFrame, df_unif: pd.DataFrame) -> pd.DataFrame:
    d = df_umap.copy()
    d["ciudad_label"]    = df_unif["ciudad"].map(CIUDADES_LABEL).fillna(df_unif["ciudad"])
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


# ─── Scatter neutral (sin colores) ───────────────────────────────────────────

def _scatter_neutral(d: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["umap1"], y=d["umap2"],
        mode="markers",
        marker=dict(size=MARKER_SIZE, color="#555555", opacity=MARKER_OPACITY),
        customdata=d[_custom_cols()].values,
        hovertemplate=_hovertemplate(),
    ))
    fig.update_layout(
        title=dict(
            text="UMAP 2D — Vista general<br>"
                 "<sup>Hover sobre un punto para ver datos reales del delito</sup>",
            font_size=14,
        ),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white",
        margin=dict(t=80, b=40, l=40, r=40),
    )
    _guardar(fig, "umap_2d_neutral.html")


# ─── Scatter por ciudad ──────────────────────────────────────────────────────

def _scatter_ciudad(d: pd.DataFrame) -> None:
    fig = px.scatter(
        d, x="umap1", y="umap2",
        color="ciudad_label",
        color_discrete_map=COLORES_CIUDAD,
        custom_data=_custom_cols(),
        title="UMAP 2D — Distribución por ciudad<br>"
              "<sup>Hover sobre un punto para ver datos reales del delito</sup>",
        labels={"umap1": "UMAP 1", "umap2": "UMAP 2",
                "ciudad_label": "Ciudad"},
        opacity=MARKER_OPACITY,
    )
    fig.update_traces(marker=dict(size=MARKER_SIZE), hovertemplate=_hovertemplate())
    _estilo(fig)
    _guardar(fig, "umap_2d_ciudad.html")


# ─── Scatter por peligrosidad ─────────────────────────────────────────────────

def _scatter_peligroso(d: pd.DataFrame) -> None:
    fig = px.scatter(
        d, x="umap1", y="umap2",
        color="peligroso_label",
        color_discrete_map=COLORES_PELIGROSO,
        custom_data=_custom_cols(),
        title="UMAP 2D — Peligroso vs No peligroso<br>"
              "<sup>Hover sobre un punto para ver datos reales del delito</sup>",
        labels={"umap1": "UMAP 1", "umap2": "UMAP 2",
                "peligroso_label": "Peligrosidad"},
        opacity=MARKER_OPACITY,
        category_orders={"peligroso_label": ["No peligroso", "Peligroso"]},
    )
    fig.update_traces(marker=dict(size=MARKER_SIZE), hovertemplate=_hovertemplate())
    _estilo(fig)
    _guardar(fig, "umap_2d_peligroso.html")


# ─── Scatter por categoría ────────────────────────────────────────────────────

def _scatter_categoria(d: pd.DataFrame) -> None:
    fig = px.scatter(
        d, x="umap1", y="umap2",
        color="categoria_label",
        custom_data=_custom_cols(),
        title="UMAP 2D — Distribución por categoría de delito<br>"
              "<sup>Hover sobre un punto para ver datos reales del delito</sup>",
        labels={"umap1": "UMAP 1", "umap2": "UMAP 2",
                "categoria_label": "Categoría"},
        opacity=MARKER_OPACITY,
    )
    fig.update_traces(marker=dict(size=MARKER_SIZE), hovertemplate=_hovertemplate())
    _estilo(fig)
    _guardar(fig, "umap_2d_categoria.html")


# ─── Hover ────────────────────────────────────────────────────────────────────

def _custom_cols() -> list:
    return ["ciudad_label", "categoria_label", "peligroso_label",
            "hora", "mes", "fecha", "latitud", "longitud"]


def _hovertemplate() -> str:
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


def _estilo(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        title_font_size=14,
        margin=dict(t=80, b=40, l=40, r=40),
    )


def _guardar(fig: go.Figure, nombre: str) -> None:
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.write_html(ruta, include_plotlyjs="cdn")
    print(f"  [html] Guardado → {ruta}")