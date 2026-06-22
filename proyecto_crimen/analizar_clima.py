"""
Script independiente para analizar la varianza y distribución
de las columnas 'viento' y 'precipitacion' del dataset unificado.

Ejecutar directamente: python analizar_clima.py
No forma parte del pipeline principal.
"""
import os
import pandas as pd
import numpy as np

RUTA_UNIFICADO = os.path.join("data", "processed", "crime_unified.csv")


def analizar_columna(df: pd.DataFrame, col: str) -> None:
    print(f"\n{'='*55}")
    print(f"  ANÁLISIS: {col.upper()}")
    print(f"{'='*55}")

    serie = df[col].dropna()
    total = len(serie)

    print(f"  Total registros      : {total:,}")
    print(f"  Media                : {serie.mean():.4f}")
    print(f"  Mediana              : {serie.median():.4f}")
    print(f"  Desviación estándar  : {serie.std():.4f}")
    print(f"  Varianza             : {serie.var():.4f}")
    print(f"  Mínimo               : {serie.min():.4f}")
    print(f"  Máximo               : {serie.max():.4f}")
    print(f"  Coef. de variación   : {(serie.std() / serie.mean() * 100 if serie.mean() != 0 else float('nan')):.2f}%")

    # Distribución de ceros
    n_ceros = (serie == 0).sum()
    pct_ceros = n_ceros / total * 100
    print(f"\n  Valores en 0         : {n_ceros:,}  ({pct_ceros:.1f}%)")

    # Percentiles
    print(f"\n  [PERCENTILES]")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = serie.quantile(p / 100)
        print(f"    P{p:<3} : {val:.4f}")

    # Distribución por rangos (para entender concentración)
    print(f"\n  [DISTRIBUCIÓN POR RANGOS]")
    bins = np.linspace(serie.min(), serie.max(), 6)
    for i in range(len(bins) - 1):
        mask = (serie >= bins[i]) & (serie < bins[i + 1])
        n = mask.sum()
        pct = n / total * 100
        print(f"    [{bins[i]:.2f} - {bins[i+1]:.2f}): {n:>8,} ({pct:>5.1f}%)")


def conclusion(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print(f"  CONCLUSIÓN RÁPIDA")
    print(f"{'='*55}")

    for col in ["precipitacion", "viento", "temperatura"]:
        if col not in df.columns:
            continue
        serie = df[col].dropna()
        cv = serie.std() / serie.mean() * 100 if serie.mean() != 0 else float("inf")
        pct_ceros = (serie == 0).sum() / len(serie) * 100

        veredicto = "BAJA varianza, candidata a descartar" if pct_ceros > 50 or abs(cv) > 200 else "Varianza aceptable"
        print(f"  {col:<15} CV={cv:>8.1f}%   %ceros={pct_ceros:>5.1f}%   → {veredicto}")


def main():
    if not os.path.exists(RUTA_UNIFICADO):
        print(f"[ERROR] No se encontró {RUTA_UNIFICADO}")
        return

    print("Cargando crime_unified.csv...")
    df = pd.read_csv(RUTA_UNIFICADO, low_memory=False)

    for col in ["viento", "precipitacion", "temperatura"]:
        if col in df.columns:
            analizar_columna(df, col)
        else:
            print(f"\n[AVISO] Columna '{col}' no encontrada en el dataset.")

    conclusion(df)


if __name__ == "__main__":
    main()