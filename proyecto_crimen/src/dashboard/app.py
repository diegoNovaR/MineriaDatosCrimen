import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Configuración ────────────────────────────────────────────────────────────
RUTA_UNIFIED   = os.path.join("data", "processed", "crime_unified.csv")
RUTA_UMAP      = os.path.join("data", "processed", "crime_umap_2d.csv")
RUTA_TSNE      = os.path.join("data", "processed", "crime_tsne_2d.csv")
RUTA_TSNE_IDX  = os.path.join("data", "processed", "crime_tsne_indices.csv")
RUTA_CLUSTERS  = os.path.join("data", "processed", "crime_clusters.csv")
RUTA_DBSCAN    = os.path.join("data", "processed", "crime_dbscan.csv")
N_MUESTRA     = 15_000   # puntos por scatter (estratificado por ciudad)
RANDOM_STATE  = 42

METODO_INFO = "UMAP  |  n_neighbors=15  |  min_dist=0.1  |  metric=euclidean  |  sin lat/lon en el vector"

COLORES_CIUDAD = {
    "Chicago":       "#2196F3",
    "Philadelphia":  "#F44336",
    "San Francisco": "#4CAF50",
}
COLORES_DBSCAN = {
    "Ruido":    "#cccccc",
    "Sin dato": "#eeeeee",
    "DBSCAN 0": "#E53935",
    "DBSCAN 1": "#1E88E5",
    "DBSCAN 2": "#43A047",
    "DBSCAN 3": "#FB8C00",
    "DBSCAN 4": "#8E24AA",
    "DBSCAN 5": "#00ACC1",
    "DBSCAN 6": "#F4511E",
    "DBSCAN 7": "#6D4C41",
    "DBSCAN 8": "#FFB300",
    "DBSCAN 9": "#3949AB",
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


def cargar_datos():
    print("Cargando datos...")
    df_unif = pd.read_csv(RUTA_UNIFIED, low_memory=False, parse_dates=["fecha"])

    # UMAP
    df_umap = pd.read_csv(RUTA_UMAP, low_memory=False) if os.path.exists(RUTA_UMAP) else pd.DataFrame()

    # t-SNE
    df_tsne, idx_tsne = pd.DataFrame(), np.array([])
    if os.path.exists(RUTA_TSNE) and os.path.exists(RUTA_TSNE_IDX):
        df_tsne  = pd.read_csv(RUTA_TSNE, low_memory=False)
        idx_tsne = pd.read_csv(RUTA_TSNE_IDX)["idx_original"].values
        print(f"  t-SNE cargado: {len(df_tsne):,} filas")

    n = min(len(df_unif), len(df_umap)) if not df_umap.empty else len(df_unif)
    df_unif = df_unif.iloc[:n].reset_index(drop=True)

    df = df_unif.copy()
    df["idx_original"] = df.index

    if not df_umap.empty:
        df["umap1"] = df_umap["umap1"].iloc[:n].values
        df["umap2"] = df_umap["umap2"].iloc[:n].values
    else:
        df["umap1"] = np.nan
        df["umap2"] = np.nan

    df["ciudad_label"]    = df["ciudad"].map(CIUDADES_LABEL).fillna(df["ciudad"])
    df["categoria_label"] = df["categoria_delito"].map(CATEGORIA_LABEL).fillna(df["categoria_delito"])
    df["peligroso_label"] = df["es_peligroso"].astype(bool).map({True: "Peligroso", False: "No peligroso"})
    df["fecha_str"]       = pd.to_datetime(df["fecha"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    import holidays
    ESTADO_POR_CIUDAD = {"chicago": "IL", "philadelphia": "PA", "san_francisco": "CA"}
    anios = df["fecha"].dt.year.dropna().unique().tolist()
    calendarios = {c: holidays.US(state=e, years=anios) for c, e in ESTADO_POR_CIUDAD.items()}
    df["es_feriado"] = df.apply(
        lambda row: int(row["fecha"].date() in calendarios.get(row["ciudad"], {}))
        if pd.notna(row["fecha"]) else 0, axis=1
    )

    if os.path.exists(RUTA_CLUSTERS):
        df_cl = pd.read_csv(RUTA_CLUSTERS)
        n_cl  = min(len(df_cl), n)
        df["cluster"]       = df_cl["cluster"].iloc[:n_cl].values
        df["cluster_label"] = "Cluster " + df["cluster"].astype(str)
    else:
        df["cluster_label"] = "Sin cluster"

    # Clusters DBSCAN
    if os.path.exists(RUTA_DBSCAN):
        df_db  = pd.read_csv(RUTA_DBSCAN)
        idx_db = df_db["idx_original"].values
        lbl_db = df_db["dbscan_label"].values
        dbscan_map = pd.Series(lbl_db, index=idx_db)
        df["dbscan_label_num"] = df["idx_original"].map(dbscan_map).fillna(-2).astype(int)
        df["dbscan_label"] = df["dbscan_label_num"].apply(
            lambda x: "Ruido" if x == -1 else ("Sin dato" if x == -2 else f"DBSCAN {x}")
        )
        n_db = (df["dbscan_label_num"] >= 0).sum()
        print(f"  DBSCAN cargado: {n_db:,} puntos en clusters")
    else:
        df["dbscan_label"] = "Sin DBSCAN"

    print(f"Datos cargados: {len(df):,} registros")
    return df, df_tsne, idx_tsne



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


DF_FULL, DF_TSNE, IDX_TSNE = cargar_datos()
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
                    {"label": "Ciudad",           "value": "ciudad_label"},
                    {"label": "Categoría",         "value": "categoria_label"},
                    {"label": "Peligrosidad",      "value": "peligroso_label"},
                    {"label": "Cluster K-Means",   "value": "cluster_label"},
                    {"label": "Cluster DBSCAN",    "value": "dbscan_label"},
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

        html.Div([
            html.Label("Método proyección:", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Dropdown(
                id="dd-metodo",
                options=[
                    {"label": "UMAP", "value": "umap"},
                    {"label": "t-SNE", "value": "tsne"},
                ],
                value="umap",
                clearable=False,
                style={"width": "140px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),

        html.Div(id="info-seleccion", style={"fontSize": "13px", "color": "#555", "marginLeft": "auto"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "24px",
              "padding": "12px 24px", "background": "#f5f5f5", "flexWrap": "wrap"}),

    # Fila principal: Scatter + Coordenadas paralelas
    html.Div([
        html.Div([
            html.H4(id="titulo-scatter", style={"margin": "0 0 4px 0", "fontSize": "14px"}),
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
            html.H4("Distribución por Hora, Mes y Día de semana",
                    style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            html.P("Distribución temporal de los puntos seleccionados",
                   style={"margin": "0 0 4px 0", "fontSize": "11px", "color": "#777"}),
            dcc.Graph(id="histogramas", style={"height": "320px"}),
        ], style={"flex": "1", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),

        html.Div([
            html.H4("Heatmap: Categoría × Hora", style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            dcc.Graph(id="heatmap", style={"height": "320px"}),
        ], style={"flex": "1.2", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),
    ], style={"display": "flex", "gap": "12px", "padding": "0 24px 12px 24px"}),

    # Fila: Temperatura media por categoría (sola)
    html.Div([
        html.Div([
            html.H4("Temperatura media por categoría de delito",
                    style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            html.P("Temperatura promedio con rango mín/máx por tipo de crimen",
                   style={"margin": "0 0 8px 0", "fontSize": "11px", "color": "#777"}),
            dcc.Graph(id="temp-categoria", style={"height": "340px"}),
        ], style={"flex": "1", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),
    ], style={"display": "flex", "gap": "12px", "padding": "0 24px 12px 24px"}),

    # Fila: Heatmap temperatura + Heatmap viento (juntos)
    html.Div([
        html.Div([
            html.H4("Heatmap: Categoría × Rango de temperatura",
                    style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            html.P("Concentración de delitos según frío / templado / caliente",
                   style={"margin": "0 0 8px 0", "fontSize": "11px", "color": "#777"}),
            dcc.Graph(id="heatmap-temp-cat", style={"height": "340px"}),
        ], style={"flex": "1", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),

        html.Div([
            html.H4("Heatmap: Categoría × Rango de viento",
                    style={"margin": "0 0 4px 0", "fontSize": "14px"}),
            html.P("Concentración de delitos según calma / moderado / fuerte",
                   style={"margin": "0 0 8px 0", "fontSize": "11px", "color": "#777"}),
            dcc.Graph(id="heatmap-viento-cat", style={"height": "340px"}),
        ], style={"flex": "1", "background": "white", "borderRadius": "8px",
                  "padding": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),
    ], style={"display": "flex", "gap": "12px", "padding": "0 24px 12px 24px"}),

    # Fila: Mapas por ciudad (tabs)
    html.Div([
        html.Div([
            html.Div([
                html.H4("Mapa de ubicación de delitos por ciudad",
                        style={"margin": "0", "fontSize": "14px"}),
                html.P("Selecciona puntos en el scatter — cada tab muestra una ciudad",
                       style={"margin": "4px 0 0 0", "fontSize": "11px", "color": "#777"}),
            ]),
            html.Div([
                html.Label("Color en mapa:", style={"fontSize": "12px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="dd-mapa-color",
                    options=[
                        {"label": "Peligrosidad",  "value": "peligroso_label"},
                        {"label": "Categoría",     "value": "categoria_label"},
                        {"label": "Cluster DBSCAN","value": "dbscan_label"},
                        {"label": "K-Means",       "value": "cluster_label"},
                    ],
                    value="peligroso_label",
                    clearable=False,
                    style={"width": "160px", "fontSize": "12px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "12px"}),

        dcc.Tabs([
            dcc.Tab(label="🔵 Chicago", children=[
                dcc.Graph(id="mapa-chicago", style={"height": "420px"}),
            ], style={"fontWeight": "bold", "color": "#2196F3"},
               selected_style={"fontWeight": "bold", "color": "#2196F3",
                               "borderTop": "3px solid #2196F3"}),

            dcc.Tab(label="🔴 Philadelphia", children=[
                dcc.Graph(id="mapa-philadelphia", style={"height": "420px"}),
            ], style={"fontWeight": "bold", "color": "#F44336"},
               selected_style={"fontWeight": "bold", "color": "#F44336",
                               "borderTop": "3px solid #F44336"}),

            dcc.Tab(label="🟢 San Francisco", children=[
                dcc.Graph(id="mapa-sanfrancisco", style={"height": "420px"}),
            ], style={"fontWeight": "bold", "color": "#4CAF50"},
               selected_style={"fontWeight": "bold", "color": "#4CAF50",
                               "borderTop": "3px solid #4CAF50"}),
        ], style={"fontFamily": "Segoe UI, Arial, sans-serif"}),

    ], style={"background": "white", "borderRadius": "8px", "padding": "16px",
              "margin": "0 24px 12px 24px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"}),

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
    Output("scatter-umap",   "figure"),
    Output("titulo-scatter", "children"),
    Input("dd-color",   "value"),
    Input("dd-ciudad",  "value"),
    Input("slider-size","value"),
    Input("dd-metodo",  "value"),
)
def actualizar_scatter(color_col, dd_ciudad, punto_size, metodo):
    if metodo == "tsne" and len(DF_TSNE) > 0:
        df_base = DF_FULL.iloc[IDX_TSNE].copy().reset_index(drop=True)
        df_base["umap1"] = DF_TSNE["tsne1"].values
        df_base["umap2"] = DF_TSNE["tsne2"].values
        eje_x, eje_y    = "t-SNE 1", "t-SNE 2"
        titulo_metodo   = "t-SNE 2D  |  perplexity=30  |  n_iter=1000"
        titulo_panel    = "Proyección t-SNE 2D"
    else:
        df_base       = DF_SAMPLE.copy()
        eje_x, eje_y  = "UMAP 1", "UMAP 2"
        titulo_metodo = "UMAP 2D  |  n_neighbors=15  |  min_dist=0.1"
        titulo_panel  = "Proyección UMAP 2D"

    if dd_ciudad != "todas":
        df_base = df_base[df_base["ciudad_label"] == dd_ciudad]

    if color_col == "ciudad_label":
        color_map = COLORES_CIUDAD
    elif color_col == "peligroso_label":
        color_map = COLORES_PELIGROSO
    elif color_col == "dbscan_label":
        color_map = COLORES_DBSCAN
    elif color_col == "cluster_label":
        color_map = None
    else:
        color_map = None

    label_map = {
        "ciudad_label":    "Ciudad",
        "categoria_label": "Categoría",
        "peligroso_label": "Peligrosidad",
        "cluster_label":   "Cluster K-Means",
        "dbscan_label":    "Cluster DBSCAN",
    }

    fig = px.scatter(
        df_base, x="umap1", y="umap2",
        color=color_col,
        color_discrete_map=color_map,
        custom_data=["idx_original", "ciudad_label", "categoria_label",
                     "peligroso_label", "hora", "mes", "fecha_str",
                     "latitud", "longitud", "temperatura", "cluster_label",
                     "dbscan_label"],
        labels={"umap1": eje_x, "umap2": eje_y,
                color_col: label_map.get(color_col, color_col)},
        opacity=0.5,
        title=titulo_metodo,
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
            "<b>K-Means</b>: %{customdata[10]}<br>"
            "<b>DBSCAN</b>: %{customdata[11]}<br>"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(title=label_map.get(color_col, color_col), font=dict(size=11)),
        dragmode="select",
        clickmode="event+select",
        uirevision="scatter",
    )
    return fig, titulo_panel


@app.callback(
    Output("histogramas",        "figure"),
    Output("heatmap",            "figure"),
    Output("parallel-coords",    "figure"),
    Output("temp-categoria",     "figure"),
    Output("heatmap-temp-cat",   "figure"),
    Output("heatmap-viento-cat", "figure"),
    Output("tabla-comparativa",  "data"),
    Output("tabla-comparativa",  "columns"),
    Output("info-seleccion",     "children"),
    Input("scatter-umap",  "selectedData"),
    Input("scatter-umap",  "clickData"),
    Input("dd-ciudad",     "value"),
    Input("dd-metodo",     "value"),
)
def actualizar_vistas(selected_data, click_data, dd_ciudad, metodo):
    if metodo == "tsne" and len(DF_TSNE) > 0:
        df_base = DF_FULL.iloc[IDX_TSNE].copy().reset_index(drop=True)
        df_base["umap1"] = DF_TSNE["tsne1"].values
        df_base["umap2"] = DF_TSNE["tsne2"].values
    else:
        df_base = DF_SAMPLE.copy()

    if dd_ciudad != "todas":
        df_base = df_base[df_base["ciudad_label"] == dd_ciudad]

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

    # ── Histogramas hora, mes y día de semana ──────────────────────────────────
    ORDEN_DIAS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    LABEL_DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

    fig_hist = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Hora del día", "Mes del año", "Día de la semana"]
    )
    for ciudad, color in COLORES_CIUDAD.items():
        sub = df_sel[df_sel["ciudad_label"] == ciudad]
        if sub.empty:
            continue
        # Hora
        fig_hist.add_trace(go.Histogram(
            x=sub["hora"], name=ciudad, marker_color=color,
            opacity=0.7, nbinsx=24, showlegend=True), row=1, col=1)
        # Mes
        fig_hist.add_trace(go.Histogram(
            x=sub["mes"], name=ciudad, marker_color=color,
            opacity=0.7, nbinsx=12, showlegend=False), row=1, col=2)
        # Día de semana
        if "dia_semana" in sub.columns:
            conteo_dia = sub["dia_semana"].str.lower().value_counts()
            conteo_ord = [conteo_dia.get(d, 0) for d in ORDEN_DIAS]
            fig_hist.add_trace(go.Bar(
                x=LABEL_DIAS, y=conteo_ord, name=ciudad,
                marker_color=color, opacity=0.7, showlegend=False), row=1, col=3)

    fig_hist.update_layout(
        margin=dict(t=35, b=20, l=20, r=20),
        barmode="overlay",
        legend=dict(font=dict(size=10)),
        height=300,
    )

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

    # ── Heatmap temperatura × categoría ────────────────────────────────────
    if "temperatura" in df_sel.columns and "categoria_label" in df_sel.columns:
        df_temp_cat = df_sel[["temperatura", "categoria_label"]].dropna().copy()

        # Clasificar temperatura en rangos
        bins   = [-30, 5, 15, 25, 50]
        labels = ["Frío (<5°C)", "Templado (5-15°C)", "Cálido (15-25°C)", "Caliente (>25°C)"]
        df_temp_cat["rango_temp"] = pd.cut(
            df_temp_cat["temperatura"], bins=bins, labels=labels
        )
        pivot_tc = (df_temp_cat
                    .groupby(["categoria_label", "rango_temp"], observed=True)
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=labels, fill_value=0))

        fig_heat_tc = px.imshow(
            pivot_tc,
            color_continuous_scale="Blues",
            labels=dict(x="Rango de temperatura", y="Categoría", color="Delitos"),
            aspect="auto",
            title="",
        )
        fig_heat_tc.update_layout(
            margin=dict(t=10, b=20, l=20, r=20),
            height=300,
            coloraxis_colorbar=dict(thickness=12, len=0.8),
        )
    else:
        fig_heat_tc = go.Figure()
        fig_heat_tc.update_layout(title="Sin datos disponibles")

    # ── Heatmap viento × categoría ──────────────────────────────────────────
    if "viento" in df_sel.columns and "categoria_label" in df_sel.columns:
        df_viento_cat = df_sel[["viento", "categoria_label"]].dropna().copy()

        bins_v   = [0, 10, 25, 40, 100]
        labels_v = ["Calma (<10)", "Moderado (10-25)", "Fuerte (25-40)", "Muy fuerte (>40)"]
        df_viento_cat["rango_viento"] = pd.cut(
            df_viento_cat["viento"], bins=bins_v, labels=labels_v
        )
        pivot_vc = (df_viento_cat
                    .groupby(["categoria_label", "rango_viento"], observed=True)
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=labels_v, fill_value=0))

        fig_heat_vc = px.imshow(
            pivot_vc,
            color_continuous_scale="Purples",
            labels=dict(x="Rango de viento (km/h)", y="Categoría", color="Delitos"),
            aspect="auto",
        )
        fig_heat_vc.update_layout(
            margin=dict(t=10, b=20, l=20, r=20),
            height=300,
            coloraxis_colorbar=dict(thickness=12, len=0.8),
        )
    else:
        fig_heat_vc = go.Figure()
        fig_heat_vc.update_layout(title="Sin datos de viento disponibles")

    # ── Tabla comparativa ───────────────────────────────────────────────────
    COLS_TABLA = ["idx_original", "ciudad_label", "categoria_label", "peligroso_label",
                  "cluster_label", "dbscan_label", "hora", "mes", "fecha_str", "es_feriado",
                  "latitud", "longitud", "temperatura", "viento", "umap1", "umap2"]
    cols_disp = [c for c in COLS_TABLA if c in df_sel.columns]
    df_tabla  = df_sel[cols_disp].round(4)

    LABELS = {
        "idx_original":    "Índice",
        "ciudad_label":    "Ciudad",
        "categoria_label": "Categoría",
        "peligroso_label": "Peligrosidad",
        "cluster_label":   "Cluster K-Means",
        "dbscan_label":    "Cluster DBSCAN",
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

    return fig_hist, fig_heat, fig_para, fig_temp, fig_heat_tc, fig_heat_vc, data, columns, info


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


# Configuración de zoom por ciudad
CIUDAD_CONFIG = {
    "Chicago":       {"lat": 41.8500, "lon": -87.6500, "zoom": 10},
    "Philadelphia":  {"lat": 39.9800, "lon": -75.1500, "zoom": 10},
    "San Francisco": {"lat": 37.7700, "lon": -122.4200, "zoom": 11},
}


def _construir_mapa(df_full_sel: pd.DataFrame, ciudad: str,
                    color_col: str) -> go.Figure:
    """Construye el mapa para una ciudad específica."""
    cfg = CIUDAD_CONFIG[ciudad]
    df_ciudad = df_full_sel[df_full_sel["ciudad_label"] == ciudad].dropna(
        subset=["latitud", "longitud"]
    )

    if color_col == "peligroso_label":
        color_map = COLORES_PELIGROSO
    elif color_col == "dbscan_label":
        color_map = COLORES_DBSCAN
    else:
        color_map = None

    label_map = {
        "categoria_label": "Categoría",
        "peligroso_label": "Peligrosidad",
        "dbscan_label":    "Cluster DBSCAN",
        "cluster_label":   "Cluster K-Means",
    }

    if df_ciudad.empty:
        fig = go.Figure()
        fig.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lat=cfg["lat"], lon=cfg["lon"]),
                        zoom=cfg["zoom"]),
            margin=dict(t=30, b=0, l=0, r=0),
            annotations=[dict(text=f"Sin datos de {ciudad} en la selección",
                              showarrow=False, font=dict(size=14),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return fig

    fig = px.scatter_mapbox(
        df_ciudad,
        lat="latitud",
        lon="longitud",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={
            "ciudad_label":    True,
            "categoria_label": True,
            "peligroso_label": True,
            "hora":            True,
            "fecha_str":       True,
            "latitud":         False,
            "longitud":        False,
        },
        zoom=cfg["zoom"],
        center={"lat": cfg["lat"], "lon": cfg["lon"]},
        labels={color_col: label_map.get(color_col, color_col)},
        opacity=0.7,
        title=f"{ciudad} — {len(df_ciudad):,} delitos",
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(t=30, b=0, l=0, r=0),
        legend=dict(font=dict(size=10)),
    )
    return fig


@app.callback(
    Output("mapa-chicago",      "figure"),
    Output("mapa-philadelphia", "figure"),
    Output("mapa-sanfrancisco", "figure"),
    Input("scatter-umap",  "selectedData"),
    Input("scatter-umap",  "clickData"),
    Input("dd-ciudad",     "value"),
    Input("dd-mapa-color", "value"),
)
def actualizar_mapas(selected_data, click_data, dd_ciudad, color_col):

    # Determinar subset según selección
    if selected_data and selected_data.get("points"):
        indices    = [p["customdata"][0] for p in selected_data["points"]]
        df_sel     = DF_FULL[DF_FULL["idx_original"].isin(indices)].copy()
    elif click_data and click_data.get("points"):
        indices    = [p["customdata"][0] for p in click_data["points"]]
        df_sel     = DF_FULL[DF_FULL["idx_original"].isin(indices)].copy()
    else:
        df_sel = DF_SAMPLE.copy()

    # Filtro global por ciudad si aplica
    if dd_ciudad != "todas":
        df_sel = df_sel[df_sel["ciudad_label"] == dd_ciudad]

    fig_chi  = _construir_mapa(df_sel, "Chicago",       color_col)
    fig_phi  = _construir_mapa(df_sel, "Philadelphia",  color_col)
    fig_sf   = _construir_mapa(df_sel, "San Francisco", color_col)

    return fig_chi, fig_phi, fig_sf


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)