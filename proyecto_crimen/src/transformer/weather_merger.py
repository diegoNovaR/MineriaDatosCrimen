import os
import pandas as pd

RUTA_PROCESSED    = os.path.join("data", "processed")
ARCHIVO_UNIFICADO = os.path.join(RUTA_PROCESSED, "crime_unified.csv")


def unir_clima(df_unificado: pd.DataFrame, datasets_clima: dict) -> pd.DataFrame:
    """
    Une datos de clima al dataset unificado por ciudad + fecha truncada a hora.
    Exporta crime_unified.csv actualizado.
    """
    print(f"\n{'='*55}")
    print(f"  UNIÓN DE DATOS CLIMÁTICOS")
    print(f"{'='*55}")
    print(f"  Filas crimen: {len(df_unificado):,}")

    # Concatenar los 3 datasets de clima en uno
    df_clima = pd.concat(datasets_clima.values(), ignore_index=True)
    print(f"  Filas clima : {len(df_clima):,}")

    # Crear columna de join en crimen: truncar fecha a hora
    df = df_unificado.copy()
    df["fecha_hora_join"] = pd.to_datetime(
        df["fecha"], errors="coerce"
    ).dt.floor("h")

    # Merge left: conserva todos los registros de crimen
    df = df.merge(
        df_clima[["ciudad", "fecha_hora_clima", "temperatura", "precipitacion", "viento"]],
        left_on=["ciudad", "fecha_hora_join"],
        right_on=["ciudad", "fecha_hora_clima"],
        how="left",
    )

    # Limpiar columnas auxiliares del join
    df = df.drop(columns=["fecha_hora_join", "fecha_hora_clima"], errors="ignore")

    print(f"  Filas resultado: {len(df):,}")

    # Reporte de nulos post-unión
    _reporte_nulos(df)

    # Exportar
    os.makedirs(RUTA_PROCESSED, exist_ok=True)
    df.to_csv(ARCHIVO_UNIFICADO, index=False, encoding="utf-8")
    print(f"\n  Exportado → {ARCHIVO_UNIFICADO}")

    return df


def _reporte_nulos(df: pd.DataFrame) -> None:
    """Reporte de nulos en las columnas de clima."""
    cols_clima = ["temperatura", "precipitacion", "viento"]
    total = len(df)

    print(f"\n  [NULOS POST-UNIÓN — COLUMNAS CLIMA]")
    print(f"  {'Columna':<20} {'Nulos':>8}  {'%':>6}")
    print(f"  {'-'*38}")

    hay_nulos = False
    for col in cols_clima:
        if col in df.columns:
            n = df[col].isnull().sum()
            pct = n / total * 100
            print(f"  {col:<20} {n:>8,}  {pct:>5.1f}%")
            if n > 0:
                hay_nulos = True

    if not hay_nulos:
        print(f"  ✔ Sin nulos en columnas de clima.")
    else:
        # Desglose por ciudad
        print(f"\n  [NULOS POR CIUDAD]")
        for ciudad in df["ciudad"].unique():
            sub = df[df["ciudad"] == ciudad]
            nulos = sub[cols_clima].isnull().sum().sum()
            pct   = nulos / (len(sub) * len(cols_clima)) * 100
            print(f"  {ciudad:<20} {nulos:>6,} nulos ({pct:.1f}%)")

        # Rango de fechas sin match
        mask_nulo = df["temperatura"].isnull()
        if mask_nulo.any():
            fechas_sin = df[mask_nulo]["fecha"]
            print(f"\n  Rango sin clima: {fechas_sin.min()} → {fechas_sin.max()}")


def analizar_nulos_clima(df_unificado: pd.DataFrame) -> None:
    """Análisis detallado de nulos en columnas de clima."""
    print(f"\n{'='*55}")
    print(f"  ANÁLISIS DE NULOS — CLIMA")
    print(f"{'='*55}")

    _reporte_nulos(df_unificado)

    cols_clima = ["temperatura", "precipitacion", "viento"]
    existentes = [c for c in cols_clima if c in df_unificado.columns]

    if not existentes:
        print("\n  [AVISO] No hay columnas de clima en el dataset.")
        return

    # Estadísticas básicas de las columnas de clima
    print(f"\n  [ESTADÍSTICAS CLIMA]")
    print(f"  {'Columna':<20} {'Media':>10} {'Mín':>10} {'Máx':>10}")
    print(f"  {'-'*52}")
    for col in existentes:
        media = df_unificado[col].mean()
        minv  = df_unificado[col].min()
        maxv  = df_unificado[col].max()
        print(f"  {col:<20} {media:>10.2f} {minv:>10.2f} {maxv:>10.2f}")