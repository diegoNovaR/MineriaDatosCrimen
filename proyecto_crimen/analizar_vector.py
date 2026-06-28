"""
Script independiente para evaluar la calidad del vector de características.
Analiza varianza, correlación, distribución, balance OHE y contribución al clustering.

Ejecutar directamente desde la raíz del proyecto:
    python analizar_vector.py
"""
import os
import pandas as pd
import numpy as np

RUTA_FEATURES  = os.path.join("data", "processed", "crime_features.csv")
RUTA_CLUSTERS  = os.path.join("data", "processed", "crime_clusters.csv")

UMBRAL_VARIANZA     = 0.01   # Variables con varianza menor serán candidatas a eliminar
UMBRAL_CORRELACION  = 0.85   # Pares con correlación mayor serán candidatos a revisar
UMBRAL_OHE_BALANCE  = 0.01   # Variables OHE con menos del 1% de unos serán revisadas


def separador(titulo: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")


# ─── Carga ────────────────────────────────────────────────────────────────────

def cargar_datos() -> tuple:
    print("\nCargando datos...")

    if not os.path.exists(RUTA_FEATURES):
        print(f"[ERROR] No existe: {RUTA_FEATURES}")
        return None, None

    df_feat = pd.read_csv(RUTA_FEATURES, low_memory=False)
    print(f"  crime_features.csv → {len(df_feat):,} filas, {df_feat.shape[1]} columnas")

    df_clust = None
    if os.path.exists(RUTA_CLUSTERS):
        df_clust = pd.read_csv(RUTA_CLUSTERS)
        print(f"  crime_clusters.csv → {len(df_clust):,} filas")
    else:
        print(f"  [AVISO] crime_clusters.csv no encontrado — se omitirá análisis 5")

    return df_feat, df_clust


# ─── Análisis 1: Varianza por columna ────────────────────────────────────────

def analizar_varianza(df: pd.DataFrame) -> None:
    separador("ANÁLISIS 1 — VARIANZA POR COLUMNA")

    varianzas = df.var().sort_values()
    bajas      = varianzas[varianzas < UMBRAL_VARIANZA]
    aceptables = varianzas[varianzas >= UMBRAL_VARIANZA]

    print(f"\n  Umbral mínimo de varianza : {UMBRAL_VARIANZA}")
    print(f"  Variables con varianza aceptable : {len(aceptables)}")
    print(f"  Variables con varianza BAJA      : {len(bajas)}")

    if not bajas.empty:
        print(f"\n  [CANDIDATAS A ELIMINAR — varianza < {UMBRAL_VARIANZA}]")
        print(f"  {'Variable':<35} {'Varianza':>12}")
        print(f"  {'-'*50}")
        for col, var in bajas.items():
            print(f"  {col:<35} {var:>12.6f}")
    else:
        print(f"\n  ✔ Ninguna variable tiene varianza crítica.")

    print(f"\n  [TOP 5 — mayor varianza]")
    print(f"  {'Variable':<35} {'Varianza':>12}")
    print(f"  {'-'*50}")
    for col, var in varianzas.tail(5).iloc[::-1].items():
        print(f"  {col:<35} {var:>12.6f}")

    print(f"\n  [BOTTOM 5 — menor varianza]")
    print(f"  {'Variable':<35} {'Varianza':>12}")
    print(f"  {'-'*50}")
    for col, var in varianzas.head(5).items():
        print(f"  {col:<35} {var:>12.6f}")


# ─── Análisis 2: Correlación entre variables ─────────────────────────────────

def analizar_correlacion(df: pd.DataFrame) -> None:
    separador("ANÁLISIS 2 — CORRELACIÓN ENTRE VARIABLES")

    print(f"\n  Umbral de correlación alta : {UMBRAL_CORRELACION}")
    print(f"  Calculando matriz de correlación...")

    corr = df.corr().abs()
    # Tomar solo triángulo superior para evitar duplicados
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pares_altos = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "Variable A", "level_1": "Variable B", 0: "Correlación"})
        .query(f"Correlación > {UMBRAL_CORRELACION}")
        .sort_values("Correlación", ascending=False)
    )

    if pares_altos.empty:
        print(f"\n  ✔ Ningún par de variables supera el umbral de correlación.")
    else:
        print(f"\n  [PARES CON CORRELACIÓN ALTA — posible redundancia]")
        print(f"  {'Variable A':<30} {'Variable B':<30} {'Correlación':>12}")
        print(f"  {'-'*75}")
        for _, row in pares_altos.iterrows():
            print(f"  {row['Variable A']:<30} {row['Variable B']:<30} {row['Correlación']:>12.4f}")


# ─── Análisis 3: Distribución de variables continuas ─────────────────────────

def analizar_distribucion(df: pd.DataFrame) -> None:
    separador("ANÁLISIS 3 — DISTRIBUCIÓN DE VARIABLES CONTINUAS ESCALADAS")

    cols_continuas = [c for c in df.columns if c.endswith("_scaled")]

    if not cols_continuas:
        print("\n  [AVISO] No se encontraron columnas _scaled")
        return

    print(f"\n  {'Variable':<25} {'Media':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Skew':>8}")
    print(f"  {'-'*65}")
    for col in cols_continuas:
        serie = df[col].dropna()
        media = serie.mean()
        std   = serie.std()
        minv  = serie.min()
        maxv  = serie.max()
        skew  = serie.skew()
        alerta = "  ← sesgo alto" if abs(skew) > 2 else ""
        print(f"  {col:<25} {media:>8.3f} {std:>8.3f} {minv:>8.3f} {maxv:>8.3f} {skew:>8.3f}{alerta}")

    print(f"\n  [Nota] Skew > 2 o < -2 indica distribución sesgada que puede afectar PCA")


# ─── Análisis 4: Balance de variables OHE ────────────────────────────────────

def analizar_balance_ohe(df: pd.DataFrame) -> None:
    separador("ANÁLISIS 4 — BALANCE DE VARIABLES OHE (cat_*, dia_*, ciudad_*)")

    cols_ohe = [c for c in df.columns
                if c.startswith(("cat_", "dia_", "ciudad_"))]

    if not cols_ohe:
        print("\n  [AVISO] No se encontraron columnas OHE")
        return

    total = len(df)
    resultados = []
    for col in cols_ohe:
        n_unos  = df[col].sum()
        pct     = n_unos / total * 100
        resultados.append((col, int(n_unos), pct))

    resultados.sort(key=lambda x: x[2])

    desbalanceadas = [(c, n, p) for c, n, p in resultados if p < UMBRAL_OHE_BALANCE * 100]
    aceptables     = [(c, n, p) for c, n, p in resultados if p >= UMBRAL_OHE_BALANCE * 100]

    print(f"\n  Umbral mínimo de balance OHE : {UMBRAL_OHE_BALANCE*100:.1f}% de unos")
    print(f"  Variables OHE aceptables     : {len(aceptables)}")
    print(f"  Variables OHE desbalanceadas : {len(desbalanceadas)}")

    if desbalanceadas:
        print(f"\n  [CANDIDATAS A REVISAR — menos del {UMBRAL_OHE_BALANCE*100:.1f}% de unos]")
        print(f"  {'Variable':<35} {'N unos':>10} {'%':>8}")
        print(f"  {'-'*56}")
        for col, n, p in desbalanceadas:
            print(f"  {col:<35} {n:>10,} {p:>8.3f}%")

    print(f"\n  [DISTRIBUCIÓN COMPLETA]")
    print(f"  {'Variable':<35} {'N unos':>10} {'%':>8}")
    print(f"  {'-'*56}")
    for col, n, p in resultados:
        marca = " ←" if p < UMBRAL_OHE_BALANCE * 100 else ""
        print(f"  {col:<35} {n:>10,} {p:>8.2f}%{marca}")


# ─── Análisis 5: Contribución al clustering ──────────────────────────────────

def analizar_contribucion_clustering(df: pd.DataFrame,
                                     df_clust: pd.DataFrame) -> None:
    separador("ANÁLISIS 5 — CONTRIBUCIÓN AL CLUSTERING (K-Means)")

    if df_clust is None:
        print("\n  [OMITIDO] crime_clusters.csv no disponible")
        return

    n = min(len(df), len(df_clust))
    df_feat   = df.iloc[:n].reset_index(drop=True).copy()
    df_feat["cluster"] = df_clust["cluster"].iloc[:n].values

    k = df_feat["cluster"].nunique()
    print(f"\n  Clusters encontrados : {k}")
    print(f"  Filas analizadas     : {n:,}")

    # Para cada variable calcular la diferencia de medias entre clusters
    # usando la varianza entre grupos (between-cluster variance)
    cols_features = [c for c in df.columns]
    separacion = {}

    global_mean = df_feat[cols_features].mean()
    for col in cols_features:
        cluster_means = df_feat.groupby("cluster")[col].mean()
        between_var   = ((cluster_means - global_mean[col]) ** 2).mean()
        separacion[col] = between_var

    sep_series = pd.Series(separacion).sort_values(ascending=False)

    print(f"\n  [TOP 10 — variables que MÁS separan los clusters]")
    print(f"  {'Variable':<35} {'Varianza entre clusters':>25}")
    print(f"  {'-'*62}")
    for col, val in sep_series.head(10).items():
        print(f"  {col:<35} {val:>25.6f}")

    print(f"\n  [BOTTOM 10 — variables que MENOS separan los clusters]")
    print(f"  {'Variable':<35} {'Varianza entre clusters':>25}")
    print(f"  {'-'*62}")
    for col, val in sep_series.tail(10).items():
        print(f"  {col:<35} {val:>25.6f}")


# ─── Resumen final ────────────────────────────────────────────────────────────

def resumen_final(df: pd.DataFrame) -> None:
    separador("RESUMEN GENERAL DEL VECTOR")

    cols_continuas = [c for c in df.columns if c.endswith("_scaled")]
    cols_binarias  = [c for c in df.columns if c in ["es_peligroso", "es_feriado"]]
    cols_cat       = [c for c in df.columns if c.startswith("cat_")]
    cols_dia       = [c for c in df.columns if c.startswith("dia_")]
    cols_ciudad    = [c for c in df.columns if c.startswith("ciudad_")]

    print(f"\n  Total columnas          : {df.shape[1]}")
    print(f"  Filas                   : {len(df):,}")
    print(f"  Nulos en el vector      : {df.isnull().sum().sum()}")
    print(f"\n  Desglose por tipo:")
    print(f"    Continuas escaladas   : {len(cols_continuas)} → {cols_continuas}")
    print(f"    Binarias directas     : {len(cols_binarias)} → {cols_binarias}")
    print(f"    OHE categoria_delito  : {len(cols_cat)} columnas")
    print(f"    OHE dia_semana        : {len(cols_dia)} columnas")
    print(f"    OHE ciudad            : {len(cols_ciudad)} columnas")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  EVALUACIÓN DEL VECTOR DE CARACTERÍSTICAS")
    print("  Crime Mining — Análisis independiente")
    print("="*60)

    df_feat, df_clust = cargar_datos()
    if df_feat is None:
        return

    resumen_final(df_feat)
    analizar_varianza(df_feat)
    analizar_correlacion(df_feat)
    analizar_distribucion(df_feat)
    analizar_balance_ohe(df_feat)
    analizar_contribucion_clustering(df_feat, df_clust)

    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN COMPLETADA")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()