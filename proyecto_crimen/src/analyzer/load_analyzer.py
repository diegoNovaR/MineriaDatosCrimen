import pandas as pd


# ─── Bounding boxes por ciudad (lat_min, lat_max, lon_min, lon_max) ──────────
BBOX_CIUDADES = {
    "chicago":       (41.60, 42.05, -87.95, -87.50),
    "philadelphia":  (39.85, 40.15, -75.30, -74.95),
    "san_francisco": (37.65, 37.95, -122.55, -122.33),
}


def analizar(datasets: dict) -> None:
    """Ejecuta el análisis completo sobre cada dataset cargado."""
    for ciudad, df in datasets.items():
        _encabezado(ciudad)
        _dimensiones(df)
        _tipos(df)
        _nulos(df)
        _estadisticas(df)
        _categoricas(df)
        _outliers_geo(ciudad, df)


# ─── Secciones ────────────────────────────────────────────────────────────────

def _encabezado(ciudad: str) -> None:
    print("\n" + "=" * 60)
    print(f"  ANÁLISIS: {ciudad.replace('_', ' ').upper()}")
    print("=" * 60)


def _dimensiones(df: pd.DataFrame) -> None:
    print(f"\n[DIMENSIONES]")
    print(f"  Filas    : {len(df):,}")
    print(f"  Columnas : {df.shape[1]}")


def _tipos(df: pd.DataFrame) -> None:
    print(f"\n[TIPOS DE DATOS]")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<35} {str(dtype)}")


def _nulos(df: pd.DataFrame) -> None:
    print(f"\n[VALORES NULOS]")
    nulos = df.isnull().sum()
    total = len(df)
    hay_nulos = False
    for col, n in nulos.items():
        if n > 0:
            pct = (n / total) * 100
            print(f"  {col:<35} {n:>7,}  ({pct:.1f}%)")
            hay_nulos = True
    if not hay_nulos:
        print("  Sin valores nulos.")


def _estadisticas(df: pd.DataFrame) -> None:
    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        return
    print(f"\n[ESTADÍSTICAS NUMÉRICAS]")
    desc = numericas.describe().T
    for col, row in desc.iterrows():
        print(f"  {col:<35} min={row['min']:>12.4f}  max={row['max']:>12.4f}  "
              f"media={row['mean']:>12.4f}  std={row['std']:>10.4f}")


def _categoricas(df: pd.DataFrame) -> None:
    COLS_CLAVE = ["tipo_delito", "descripcion", "distrito",
                  "descripcion_lugar", "resolucion", "arresto"]
    categoricas = [c for c in COLS_CLAVE if c in df.columns]
    if not categoricas:
        return
    print(f"\n[VALORES ÚNICOS — COLUMNAS CATEGÓRICAS CLAVE]")
    for col in categoricas:
        n = df[col].nunique()
        top = df[col].value_counts().head(10).to_dict()
        top_str = ", ".join(f'"{k}"({v:,})' for k, v in top.items())
        print(f"  {col:<35} únicos={n:>5}  top10: {top_str}")


def _outliers_geo(ciudad: str, df: pd.DataFrame) -> None:
    if ciudad not in BBOX_CIUDADES:
        return
    if "latitud" not in df.columns or "longitud" not in df.columns:
        return

    lat_min, lat_max, lon_min, lon_max = BBOX_CIUDADES[ciudad]

    mask_nulo = df["latitud"].isnull() | df["longitud"].isnull()
    df_geo = df[~mask_nulo]

    mask_fuera = (
        (df_geo["latitud"]  < lat_min) | (df_geo["latitud"]  > lat_max) |
        (df_geo["longitud"] < lon_min) | (df_geo["longitud"] > lon_max)
    )

    fuera = df_geo[mask_fuera]

    print(f"\n[OUTLIERS GEOGRÁFICOS]")
    print(f"  BBox esperada → lat:[{lat_min}, {lat_max}]  lon:[{lon_min}, {lon_max}]")
    print(f"  Sin coordenadas (nulos) : {mask_nulo.sum():,}")
    print(f"  Fuera de BBox           : {len(fuera):,}")

    if not fuera.empty:
        cols_mostrar = ["latitud", "longitud"]
        for col_extra in ["tipo_delito", "fecha", "id"]:
            if col_extra in fuera.columns:
                cols_mostrar.append(col_extra)
        print(f"\n  Muestra (hasta 10 registros):")
        print(fuera[cols_mostrar].head(10).to_string(index=True))