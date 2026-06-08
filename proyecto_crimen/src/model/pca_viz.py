import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.translations import CIUDADES_LABEL, CATEGORIA_LABEL

RUTA_OUTPUTS = os.path.join("outputs", "pca")

COLORES_CIUDAD = {
    "chicago":       "#2196F3",
    "philadelphia":  "#F44336",
    "san_francisco": "#4CAF50",
}
COLORES_PELIGROSO = {
    "Peligroso":     "#EF5350",
    "No peligroso":  "#42A5F5",
}


def graficar_interactivo(df_2d: pd.DataFrame, df_3d: pd.DataFrame) -> None:
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  PCA — VISUALIZACIÓN INTERACTIVA (Plotly)")
    print(f"{'='*55}")

    # Muestra para rendimiento
    n_muestra = min(30_000, len(df_2d))
    idx = np.random.default_rng(42).choice(len(df_2d), size=n_muestra, replace=False)
    d2  = df_2d.iloc[idx].copy().reset_index(drop=True)
    d3  = df_3d.iloc[idx].copy().reset_index(drop=True)

    # Columnas auxiliares para hover
    for d in [d2, d3]:
        d["ciudad_label"]    = d["ciudad"].map(CIUDADES_LABEL).fillna(d["ciudad"])
        d["categoria_label"] = d["categoria_delito"].map(CATEGORIA_LABEL).fillna(d["categoria_delito"])
        d["peligroso_label"] = d["es_peligroso"].astype(bool).map(
            {True: "Peligroso", False: "No peligroso"})
        if "fecha" in d.columns:
            d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    _scatter_2d_ciudad(d2)
    _scatter_2d_peligroso(d2)
    _scatter_3d_ciudad(d3)
    _scatter_3d_peligroso(d3)

    print(f"\n  ✔ Visualizaciones guardadas en: {RUTA_OUTPUTS}/")


# ─── 2D por ciudad ────────────────────────────────────────────────────────────

def _scatter_2d_ciudad(d: pd.DataFrame) -> None:
    fig = px.scatter(
        d, x="pc1", y="pc2",
        color="ciudad_label",
        color_discrete_map={v: COLORES_CIUDAD[k] for k, v in CIUDADES_LABEL.items()},
        custom_data=_custom_cols(d),
        title="PCA 2D — Distribución por ciudad<br>"
              "<sup>Cada punto representa un delito. Hover para ver detalle.</sup>",
        labels={"pc1": "Componente Principal 1",
                "pc2": "Componente Principal 2",
                "ciudad_label": "Ciudad"},
        opacity=0.5,
    )
    fig.update_traces(hovertemplate=_hover_template(d))
    _aplicar_estilo(fig)
    _guardar_html(fig, "pca_2d_ciudad.html")


# ─── 2D por peligrosidad ──────────────────────────────────────────────────────

def _scatter_2d_peligroso(d: pd.DataFrame) -> None:
    fig = px.scatter(
        d, x="pc1", y="pc2",
        color="peligroso_label",
        color_discrete_map=COLORES_PELIGROSO,
        custom_data=_custom_cols(d),
        title="PCA 2D — Peligroso vs No peligroso<br>"
              "<sup>Cada punto representa un delito. Hover para ver detalle.</sup>",
        labels={"pc1": "Componente Principal 1",
                "pc2": "Componente Principal 2",
                "peligroso_label": "Peligrosidad"},
        opacity=0.5,
        category_orders={"peligroso_label": ["No peligroso", "Peligroso"]},
    )
    fig.update_traces(hovertemplate=_hover_template(d))
    _aplicar_estilo(fig)
    _guardar_html(fig, "pca_2d_peligroso.html")


# ─── 3D por ciudad ────────────────────────────────────────────────────────────

def _scatter_3d_ciudad(d: pd.DataFrame) -> None:
    if "pc3" not in d.columns:
        print("  [AVISO] pc3 no disponible. Omitiendo 3D ciudad.")
        return

    fig = px.scatter_3d(
        d, x="pc1", y="pc2", z="pc3",
        color="ciudad_label",
        color_discrete_map={v: COLORES_CIUDAD[k] for k, v in CIUDADES_LABEL.items()},
        custom_data=_custom_cols(d),
        title="PCA 3D — Distribución por ciudad",
        labels={"pc1": "PC1", "pc2": "PC2", "pc3": "PC3",
                "ciudad_label": "Ciudad"},
        opacity=0.5,
    )
    fig.update_traces(hovertemplate=_hover_template(d))
    _aplicar_estilo_3d(fig)
    _guardar_html(fig, "pca_3d_ciudad.html")


# ─── 3D por peligrosidad ──────────────────────────────────────────────────────

def _scatter_3d_peligroso(d: pd.DataFrame) -> None:
    if "pc3" not in d.columns:
        print("  [AVISO] pc3 no disponible. Omitiendo 3D peligroso.")
        return

    fig = px.scatter_3d(
        d, x="pc1", y="pc2", z="pc3",
        color="peligroso_label",
        color_discrete_map=COLORES_PELIGROSO,
        custom_data=_custom_cols(d),
        title="PCA 3D — Peligroso vs No peligroso",
        labels={"pc1": "PC1", "pc2": "PC2", "pc3": "PC3",
                "peligroso_label": "Peligrosidad"},
        opacity=0.5,
        category_orders={"peligroso_label": ["No peligroso", "Peligroso"]},
    )
    fig.update_traces(hovertemplate=_hover_template(d))
    _aplicar_estilo_3d(fig)
    _guardar_html(fig, "pca_3d_peligroso.html")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _custom_cols(d: pd.DataFrame) -> list:
    """Orden fijo de columnas para customdata."""
    candidatas = ["ciudad_label", "categoria_label", "peligroso_label",
                  "hora", "mes", "fecha", "latitud", "longitud"]
    return [c for c in candidatas if c in d.columns]


def _hover_template(d: pd.DataFrame) -> str:
    """Genera hovertemplate en base a las columnas disponibles."""
    candidatas = [
        ("ciudad_label",    "Ciudad"),
        ("categoria_label", "Categoría delito"),
        ("peligroso_label", "Peligrosidad"),
        ("hora",            "Hora"),
        ("mes",             "Mes"),
        ("fecha",           "Fecha"),
        ("latitud",         "Latitud"),
        ("longitud",        "Longitud"),
    ]
    disponibles = [c for c in candidatas if c[0] in d.columns]
    lineas = [f"<b>{label}</b>: %{{customdata[{i}]}}"
              for i, (_, label) in enumerate(disponibles)]
    return "<br>".join(lineas) + "<extra></extra>"


def _aplicar_estilo(fig: go.Figure) -> None:
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        template="plotly_white",
        title_font_size=14,
        legend_title_font_size=11,
        margin=dict(t=80, b=40, l=40, r=40),
    )
    fig.update_xaxes(showgrid=True, zeroline=True, zerolinecolor="lightgray")
    fig.update_yaxes(showgrid=True, zeroline=True, zerolinecolor="lightgray")


def _aplicar_estilo_3d(fig: go.Figure) -> None:
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        template="plotly_white",
        title_font_size=14,
        margin=dict(t=80, b=0, l=0, r=0),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
    )


def _guardar_html(fig: go.Figure, nombre: str) -> None:
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.write_html(ruta, include_plotlyjs="cdn")
    print(f"  [html] Guardado → {ruta}")