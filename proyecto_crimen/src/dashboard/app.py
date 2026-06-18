import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Configuración ────────────────────────────────────────────────────────────
RUTA_UNIFIED  = os.path.join("data", "processed", "crime_unified.csv")
RUTA_UMAP     = os.path.join("data", "processed", "crime_umap_2d.csv")
N_MUESTRA     = 90_000   # puntos por scatter (estratificado por ciudad)
RANDOM_STATE  = 42

METODO_INFO = "UMAP  |  n_neighbors=15  |  min_dist=0.1  |  metric=euclidean  |  sin lat/lon en el vector"

COLORES_CIUDAD = {
    "Chicago":       "#2196F3",
    "Philadelphia":  "#F44336",
    "San Francisco": "#4CAF50",
}
COLORES_PELIGROSO = {
    "Peligroso":    "#EF5350",
    "No peligroso": "#42A5F5",
}

CIUDADES_LABEL = {
    "chicago":       "Chicago",
    "philadelphia":  "Philadelphia",
    "san_francisco": "San Francisco",
}
CATEGORIA_LABEL = {
    "robo_simple":           "Robo simple",
    "agresion":              "Agresión",
    "daño_propiedad":        "Daño a propiedad",
    "robo_vehiculo":         "Robo de vehículo",
    "fraude_engaño":         "Fraude / Engaño",
    "robo_con_violencia":    "Robo con violencia",
    "allanamiento":          "Allanamiento",
    "drogas":                "Drogas",
    "violacion_armas":       "Violación de armas",
    "agresion_sexual":       "Agresión sexual",
    "delito_contra_menores": "Delito contra menores",
    "homicidio":             "Homicidio",
    "arson":                 "Incendio provocado",
    "amenaza_acoso":         "Amenaza / Acoso",
    "trata_personas":        "Trata de personas",
    "orden_publico":         "Orden público",
    "sin_relevancia":        "Sin clasificación",
    "otros":                 "Otros",
}

# ─── Carga de datos ───────────────────────────────────────────────────────────

def cargar_datos():
    print("Cargando datos...")
    df_unif = pd.read_csv(RUTA_UNIFIED, low_memory=False, parse_dates=["fecha"])
    df_umap = pd.read_csv(RUTA_UMAP, low_memory=False)

    n = min(len(df_unif), len(df_umap))
    df_unif = df_unif.iloc[:n].reset_index(drop=True)
    df_umap = df_umap.iloc[:n].reset_index(drop=True)

    df = pd.concat([df_umap, df_unif], axis=1)
    df["idx_original"]   = df.index
    df["ciudad_label"]   = df["ciudad"].map(CIUDADES_LABEL).fillna(df["ciudad"])
    df["categoria_label"]= df["categoria_delito"].map(CATEGORIA_LABEL).fillna(df["categoria_delito"])
    df["peligroso_label"]= df["es_peligroso"].astype(bool).map({True: "Peligroso", False: "No peligroso"})
    df["fecha_str"]      = pd.to_datetime(df["fecha"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    # Calcular es_feriado (igual que en feature_engineering.py)
    import holidays
    ESTADO_POR_CIUDAD = {"chicago": "IL", "philadelphia": "PA", "san_francisco": "CA"}
    anios = df["fecha"].dt.year.dropna().unique().tolist()
    calendarios = {c: holidays.US(state=e, years=anios) for c, e in ESTADO_POR_CIUDAD.items()}
    df["es_feriado"] = df.apply(
        lambda row: int(row["fecha"].date() in calendarios.get(row["ciudad"], {}))
        if pd.notna(row["fecha"]) else 0,
        axis=1
    )

    print(f"Datos cargados: {len(df):,} registros")
    return df


def muestra_estratificada(df):
    ciudades     = df["ciudad_label"].unique()
    n_por_ciudad = N_MUESTRA // len(ciudades)
    frames = []
    rng = np.random.default_rng(RANDOM_STATE)
    for ciudad in ciudades:
        sub = df[df["ciudad_label"] == ciudad]
        n   = min(n_por_ciudad, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        frames.append(sub.iloc[idx])
    return pd.concat(frames, ignore_index=True)


DF_FULL   = cargar_datos()
DF_SAMPLE = muestra_estratificada(DF_FULL)

# ─── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Crime Mining Dashboard")
server = app.server  # para deploy en Render

# ─── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.H2("Sistema de Análisis de Crimen Urbano", style={"margin": "0", "color": "white"}),
        html.P(f"Método: {METODO_INFO}", style={"margin": "4px 0 0 0", "color": "#ccc", "fontSize": "12px"}),
    ], style={"background": "#1a1a2e", "padding": "16px 24px"}),

    # Controles
    html.Div([
        html.Div([
            html.Label("Color del scatter:", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Dropdown(
                id="dd-color",
                options=[
                    {"label": "Ciudad",       "value": "ciudad_label"},
                    {"label": "Categoría",    "value": "categoria_label"},
                    {"label": "Peligrosidad", "value": "peligroso_label"},
                ],
                value="ciudad_label",
                clearable=False,
                style={"width": "200px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),

        html.Div([
            html.Label("Filtrar ciudad:", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Dropdown(
                id="dd-ciudad",
                options=[{"label": "Todas", "value": "todas"}] +
                        [{"label": v, "value": v} for v in sorted(DF_SAMPLE["ciudad_label"].unique())],
                value="todas",
                clearable=False,
                style={"width": "200px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),

        html.Div([
            html.Label("Tamaño puntos:", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Slider(
                id="slider-size",
                min=1, max=8, step=0.5, value=3,
                marks={1: "1", 4: "4", 8: "8"},
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px", "width": "200px"}),

        html.Div(id="info-seleccion", style={"fontSize": "13px", "color": "#555", "marginLeft": "auto"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "24px",
              "padding": "12px 24px", "background": "#f5f5f5", "flexWrap": "wrap"}),

    # Fila principal: Scatter + Coordenadas paralelas
    html.Div([
        html.Div([
            html.H4("Proyección UMAP 2D", style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            html.P("Selecciona puntos con Box Select o Lasso para coordinar las vistas",
                   style={"margin": "0 0 8px 0", "fontSize": "11px", "color": "#777"}),
            dcc.Graph(id="scatter-umap", style={"height": "420px"},
                      config={"modeBarButtonsToAdd": ["select2d", "lasso2d"]}),
        ], style={"flex": "1.2", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),

        html.Div([
            html.H4("Coordenadas Paralelas", style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            html.P("Refleja el perfil de los puntos seleccionados",
                   style={"margin": "0 0 8px 0", "fontSize": "11px", "color": "#777"}),
            dcc.Graph(id="parallel-coords", style={"height": "420px"}),
        ], style={"flex": "1", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),
    ], style={"display": "flex", "gap": "12px", "padding": "12px 24px"}),

    # Fila: Histogramas + Heatmap
    html.Div([
        html.Div([
            html.H4("Distribución por Hora y Mes", style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            dcc.Graph(id="histogramas", style={"height": "300px"}),
        ], style={"flex": "1", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),

        html.Div([
            html.H4("Heatmap: Categoría × Hora", style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            dcc.Graph(id="heatmap", style={"height": "300px"}),
        ], style={"flex": "1.2", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),
    ], style={"display": "flex", "gap": "12px", "padding": "0 24px 12px 24px"}),

    # Fila: Temperatura por categoría
    html.Div([
        html.Div([
            html.H4("Temperatura media por categoría de delito",
                    style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            html.P("Relación entre condición climática y tipo de crimen",
                   style={"margin": "0 0 8px 0", "fontSize": "11px", "color": "#777"}),
            dcc.Graph(id="temp-categoria", style={"height": "320px"}),
        ], style={"flex": "1", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),
    ], style={"display": "flex", "gap": "12px", "padding": "0 24px 12px 24px"}),

    # Tabla comparativa
    html.Div([
        html.Div([
            html.H4("Tabla comparativa de puntos seleccionados",
                    style={"margin": "0", "fontSize": "14px"}),
            html.Div([
                html.Label("Filas por página:", style={"fontSize": "12px", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="dd-page-size",
                    options=[
                        {"label": "10",   "value": 10},
                        {"label": "25",   "value": 25},
                        {"label": "50",   "value": 50},
                        {"label": "Todos","value": 99999},
                    ],
                    value=10,
                    clearable=False,
                    style={"width": "100px", "fontSize": "12px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "8px"}),
        html.P("Selecciona puntos en el scatter para ver sus valores originales",
               style={"margin": "0 0 8px 0", "fontSize": "11px", "color": "#777"}),
        dash_table.DataTable(
            id="tabla-comparativa",
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px", "padding": "6px 10px", "textAlign": "left"},
            style_header={"fontWeight": "bold", "background": "#1a1a2e", "color": "white"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
            ],
            page_size=10,
            page_action="native",
        ),
    ], style={"background": "white", "borderRadius": "8px", "padding": "16px",
              "margin": "0 24px 24px 24px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),

], style={"fontFamily": "Segoe UI, Arial, sans-serif", "background": "#f0f2f5", "minHeight": "100vh"})


# ─── Callbacks ────────────────────────────────────────────────────────────────

def _filtrar(dd_ciudad):
    if dd_ciudad == "todas":
        return DF_SAMPLE
    return DF_SAMPLE[DF_SAMPLE["ciudad_label"] == dd_ciudad]


@app.callback(
    Output("scatter-umap", "figure"),
    Input("dd-color", "value"),
    Input("dd-ciudad", "value"),
    Input("slider-size", "value"),
)
def actualizar_scatter(color_col, dd_ciudad, punto_size):
    df = _filtrar(dd_ciudad)

    if color_col == "ciudad_label":
        color_map = COLORES_CIUDAD
    elif color_col == "peligroso_label":
        color_map = COLORES_PELIGROSO
    else:
        color_map = None

    label_map = {
        "ciudad_label":    "Ciudad",
        "categoria_label": "Categoría",
        "peligroso_label": "Peligrosidad",
    }

    fig = px.scatter(
        df, x="umap1", y="umap2",
        color=color_col,
        color_discrete_map=color_map,
        custom_data=["idx_original", "ciudad_label", "categoria_label",
                     "peligroso_label", "hora", "mes", "fecha_str",
                     "latitud", "longitud", "temperatura"],
        labels={"umap1": "UMAP 1", "umap2": "UMAP 2", color_col: label_map.get(color_col, color_col)},
        opacity=0.5,
    )
    fig.update_traces(
        marker=dict(size=punto_size),
        hovertemplate=(
            "<b>Índice</b>: %{customdata[0]}<br>"
            "<b>Ciudad</b>: %{customdata[1]}<br>"
            "<b>Categoría</b>: %{customdata[2]}<br>"
            "<b>Peligrosidad</b>: %{customdata[3]}<br>"
            "<b>Hora</b>: %{customdata[4]}<br>"
            "<b>Mes</b>: %{customdata[5]}<br>"
            "<b>Fecha</b>: %{customdata[6]}<br>"
            "<b>Latitud</b>: %{customdata[7]}<br>"
            "<b>Longitud</b>: %{customdata[8]}<br>"
            "<b>Temperatura</b>: %{customdata[9]}°C<br>"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(title=label_map.get(color_col, color_col), font=dict(size=11)),
        dragmode="select",
        clickmode="event+select",
        uirevision="scatter",
    )
    return fig


@app.callback(
    Output("histogramas",      "figure"),
    Output("heatmap",          "figure"),
    Output("parallel-coords",  "figure"),
    Output("temp-categoria",   "figure"),
    Output("tabla-comparativa","data"),
    Output("tabla-comparativa","columns"),
    Output("info-seleccion",   "children"),
    Input("scatter-umap", "selectedData"),
    Input("scatter-umap", "clickData"),
    Input("dd-ciudad",    "value"),
)
def actualizar_vistas(selected_data, click_data, dd_ciudad):
    df_base = _filtrar(dd_ciudad)

    # Priorizar selección brush; si no hay, usar click individual con shift
    if selected_data and selected_data.get("points"):
        indices = [p["customdata"][0] for p in selected_data["points"]]
        df_sel  = DF_FULL[DF_FULL["idx_original"].isin(indices)].copy()
        info    = f"✔ {len(df_sel):,} puntos seleccionados"
    elif click_data and click_data.get("points"):
        indices = [p["customdata"][0] for p in click_data["points"]]
        df_sel  = DF_FULL[DF_FULL["idx_original"].isin(indices)].copy()
        info    = f"✔ {len(df_sel):,} punto(s) seleccionado(s) — Shift+Click para agregar"
    else:
        df_sel = df_base.copy()
        info   = f"Mostrando muestra: {len(df_base):,} puntos (sin selección)"

    # ── Histogramas hora y mes ──────────────────────────────────────────────
    fig_hist = make_subplots(rows=1, cols=2,
                             subplot_titles=["Distribución por Hora", "Distribución por Mes"])
    for ciudad, color in COLORES_CIUDAD.items():
        sub = df_sel[df_sel["ciudad_label"] == ciudad]
        if sub.empty:
            continue
        fig_hist.add_trace(go.Histogram(x=sub["hora"], name=ciudad, marker_color=color,
                                        opacity=0.7, nbinsx=24, showlegend=True), row=1, col=1)
        fig_hist.add_trace(go.Histogram(x=sub["mes"],  name=ciudad, marker_color=color,
                                        opacity=0.7, nbinsx=12, showlegend=False), row=1, col=2)
    fig_hist.update_layout(margin=dict(t=30, b=20, l=20, r=20), barmode="overlay",
                           legend=dict(font=dict(size=10)), height=280)

    # ── Heatmap categoría × hora ────────────────────────────────────────────
    pivot = (df_sel.groupby(["categoria_label", "hora"])
             .size().unstack(fill_value=0))
    fig_heat = px.imshow(pivot, color_continuous_scale="YlOrRd",
                         labels=dict(x="Hora", y="Categoría", color="Delitos"),
                         aspect="auto")
    fig_heat.update_layout(margin=dict(t=10, b=20, l=20, r=20), height=280,
                           coloraxis_colorbar=dict(thickness=12, len=0.8))

    # ── Coordenadas paralelas ───────────────────────────────────────────────
    df_para = df_sel[["hora", "mes", "temperatura", "es_feriado",
                       "viento", "es_peligroso"]].copy()
    df_para["es_peligroso"] = df_para["es_peligroso"].astype(int)
    df_para["es_feriado"]   = df_para["es_feriado"].astype(int)
    df_para["ciudad_num"]   = df_sel["ciudad_label"].map(
        {"Chicago": 0, "Philadelphia": 1, "San Francisco": 2}).fillna(-1)

    dims = [
        dict(label="Hora",         values=df_para["hora"],         range=[0, 23]),
        dict(label="Mes",          values=df_para["mes"],          range=[1, 12]),
        dict(label="Temperatura",  values=df_para["temperatura"]),
        dict(label="Feriado",      values=df_para["es_feriado"],   range=[0, 1],
             tickvals=[0, 1], ticktext=["No", "Sí"]),
        dict(label="Viento",       values=df_para["viento"]),
        dict(label="Peligroso",    values=df_para["es_peligroso"], range=[0, 1],
             tickvals=[0, 1], ticktext=["No", "Sí"]),
        dict(label="Ciudad",       values=df_para["ciudad_num"],   range=[0, 2],
             tickvals=[0, 1, 2], ticktext=["Chicago", "Phila.", "SF"]),
    ]
    fig_para = go.Figure(go.Parcoords(
        line=dict(color=df_para["ciudad_num"],
                  colorscale=[[0, "#2196F3"], [0.5, "#F44336"], [1, "#4CAF50"]],
                  showscale=False),
        dimensions=dims,
    ))
    fig_para.update_layout(margin=dict(t=60, b=20, l=80, r=40), height=400)

    # ── Temperatura por categoría ────────────────────────────────────────────
    if "temperatura" in df_sel.columns and "categoria_label" in df_sel.columns:
        temp_cat = (df_sel.groupby("categoria_label")["temperatura"]
                    .agg(["mean", "min", "max"])
                    .reset_index()
                    .sort_values("mean", ascending=True))
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Bar(
            y=temp_cat["categoria_label"],
            x=temp_cat["mean"],
            orientation="h",
            marker=dict(
                color=temp_cat["mean"],
                colorscale="RdYlBu_r",
                showscale=True,
                colorbar=dict(title="°C", thickness=12),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=temp_cat["max"] - temp_cat["mean"],
                arrayminus=temp_cat["mean"] - temp_cat["min"],
                color="#888", thickness=1.5,
            ),
            hovertemplate="<b>%{y}</b><br>Media: %{x:.1f}°C<extra></extra>",
        ))
        fig_temp.update_layout(
            margin=dict(t=10, b=30, l=20, r=60),
            height=300,
            xaxis_title="Temperatura media (°C)",
            yaxis_title="",
            showlegend=False,
        )
    else:
        fig_temp = go.Figure()
        fig_temp.update_layout(title="Sin datos de temperatura disponibles")

    # ── Tabla comparativa ───────────────────────────────────────────────────
    COLS_TABLA = ["idx_original", "ciudad_label", "categoria_label", "peligroso_label",
                  "hora", "mes", "fecha_str", "es_feriado", "latitud", "longitud",
                  "temperatura", "viento", "umap1", "umap2"]
    cols_disp = [c for c in COLS_TABLA if c in df_sel.columns]
    df_tabla  = df_sel[cols_disp].round(4)

    LABELS = {
        "idx_original":    "Índice",
        "ciudad_label":    "Ciudad",
        "categoria_label": "Categoría",
        "peligroso_label": "Peligrosidad",
        "hora":            "Hora",
        "mes":             "Mes",
        "fecha_str":       "Fecha",
        "es_feriado":      "Feriado",
        "latitud":         "Latitud",
        "longitud":        "Longitud",
        "temperatura":     "Temp (°C)",
        "viento":          "Viento (km/h)",
        "umap1":           "UMAP 1",
        "umap2":           "UMAP 2",
    }
    columns = [{"name": LABELS.get(c, c), "id": c} for c in cols_disp]
    data    = df_tabla.to_dict("records")

    return fig_hist, fig_heat, fig_para, fig_temp, data, columns, info


@app.callback(
    Output("tabla-comparativa", "page_size"),
    Input("dd-page-size", "value"),
    Input("tabla-comparativa", "data"),
)
def actualizar_page_size(page_size, data):
    n_filas = len(data) if data else 0
    if page_size == 99999:
        return max(n_filas, 1)
    return page_size


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)