import os
import pandas as pd

RUTAS_CLIMA = {
    "chicago":       os.path.join("data", "raw", "climaChicago.csv"),
    "philadelphia":  os.path.join("data", "raw", "climaPhilla.csv"),
    "san_francisco": os.path.join("data", "raw", "climaSF.csv"),
}

COLUMNAS_RENAME = {
    "time":                         "fecha_hora_clima",
    "temperature_2m (°C)":          "temperatura",
    "precipitation (mm)":           "precipitacion",
    "wind_speed_100m (km/h)":       "viento",
}


def cargar_clima() -> dict:
    """
    Carga los 3 CSVs de clima y retorna dict ciudad -> DataFrame.
    Parsea fecha y renombra columnas al español.
    """
    datasets_clima = {}

    for ciudad, ruta in RUTAS_CLIMA.items():
        if not os.path.exists(ruta):
            print(f"  [ERROR] No se encontró: {ruta}")
            continue

        df = pd.read_csv(ruta)
        df = df.rename(columns=COLUMNAS_RENAME)

        # Parsear fecha y truncar a hora
        df["fecha_hora_clima"] = pd.to_datetime(
            df["fecha_hora_clima"], errors="coerce"
        ).dt.floor("h")

        df["ciudad"] = ciudad
        datasets_clima[ciudad] = df

        print(f"  [clima] {ciudad:<18} → {len(df):,} registros  "
              f"({df['fecha_hora_clima'].min()} a {df['fecha_hora_clima'].max()})")

    print(f"\n  ✔ Clima cargado: {list(datasets_clima.keys())}")
    return datasets_clima