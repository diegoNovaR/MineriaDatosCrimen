import os
import pandas as pd

RUTA_PROCESADOS = os.path.join("data", "processed")


def exportar(datasets_transformados: dict) -> None:
    """Guarda cada dataset transformado como CSV en data/processed/."""
    os.makedirs(RUTA_PROCESADOS, exist_ok=True)

    for ciudad, df in datasets_transformados.items():
        ruta = os.path.join(RUTA_PROCESADOS, f"{ciudad}_transformado.csv")
        df.to_csv(ruta, index=False, encoding="utf-8")
        print(f"  [export] {ciudad} → {ruta}  ({len(df):,} filas)")

    print(f"\n  ✔ Exportación completada en: {RUTA_PROCESADOS}/")


def cargar_transformados() -> dict:
    """Carga los CSVs transformados desde data/processed/ directamente."""
    datasets = {}

    if not os.path.exists(RUTA_PROCESADOS):
        print(f"\n[ERROR] No existe la carpeta: {RUTA_PROCESADOS}")
        return datasets

    archivos = [f for f in os.listdir(RUTA_PROCESADOS) if f.endswith("_transformado.csv")]

    if not archivos:
        print(f"\n[AVISO] No hay archivos transformados en {RUTA_PROCESADOS}/")
        return datasets

    for archivo in sorted(archivos):
        ciudad = archivo.replace("_transformado.csv", "")
        ruta = os.path.join(RUTA_PROCESADOS, archivo)

        df = pd.read_csv(
            ruta,
            low_memory=False,
            parse_dates=["fecha"],
        )

        # Restaurar tipos
        for col in ["hora", "mes", "anio", "distrito"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        df["es_fin_semana"] = df["es_fin_semana"].astype(bool)
        df["es_peligroso"]  = df["es_peligroso"].astype(bool)

        datasets[ciudad] = df
        print(f"  [carga] {ciudad} ← {ruta}  ({len(df):,} filas)")

    print(f"\n  ✔ Datasets transformados cargados: {list(datasets.keys())}")
    return datasets