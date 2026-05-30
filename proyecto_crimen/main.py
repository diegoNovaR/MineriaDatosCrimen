import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.loader.chicago_loader import ChicagoLoader
from src.loader.phila_loader import PhilaLoader
from src.loader.sf_loader import SFLoader
from src.analyzer import load_analyzer, column_comparator
from src.analyzer.correlation_analyzer import analizar_correlacion
from src.analyzer.hypothesis_viz import generar_todas
from src.analyzer.granularity_analyzer import analizar_granularidad
from src.analyzer.eda_analyzer import ejecutar_eda
from src.model.clustering import ejecutar_clustering
from src.model.feature_engineering import unificar, generar_features, cargar_unificado, cargar_features
from src.dashboard.dashboard_generator import generar_dashboard
from src.cleaner import cleaner_pipeline
from src.transformer import transformer_pipeline
from src.transformer.exporter import exportar, cargar_transformados

# ─── Rutas ───────────────────────────────────────────────────────────────────
RUTAS = {
    "chicago":       os.path.join("data", "raw", "ChicagoData.csv"),
    "philadelphia":  os.path.join("data", "raw", "PhilaData.csv"),
    "san_francisco": os.path.join("data", "raw", "SanFranciscoData.csv"),
}

LOADERS = {
    "chicago":       ChicagoLoader,
    "philadelphia":  PhilaLoader,
    "san_francisco": SFLoader,
}


# ─── Carga raw ────────────────────────────────────────────────────────────────

def cargar_todos(anio: int = 2025) -> dict:
    datasets = {}
    for ciudad, Loader in LOADERS.items():
        ruta = RUTAS[ciudad]
        if not os.path.exists(ruta):
            print(f"\n[ERROR] No se encontró: {ruta}")
            continue
        loader = Loader(ruta)
        datasets[ciudad] = loader.cargar(anio=anio)
    return datasets


def cargar_individual(anio: int = 2025) -> dict:
    print("\n  Selecciona la ciudad:")
    ciudades = list(LOADERS.keys())
    for i, c in enumerate(ciudades, 1):
        print(f"    {i}. {c.replace('_', ' ').title()}")

    opcion = input("\n  Opción: ").strip()
    try:
        ciudad = ciudades[int(opcion) - 1]
    except (ValueError, IndexError):
        print("  [ERROR] Opción inválida.")
        return {}

    ruta = RUTAS[ciudad]
    if not os.path.exists(ruta):
        print(f"\n[ERROR] No se encontró: {ruta}")
        return {}

    loader = LOADERS[ciudad](ruta)
    return {ciudad: loader.cargar(anio=anio)}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _verificar(datasets: dict, msg: str = "") -> bool:
    if not datasets:
        print(f"\n  [AVISO] {msg}" if msg else "\n  [AVISO] No hay datasets disponibles.")
        return False
    return True


# ─── Menú ─────────────────────────────────────────────────────────────────────

def menu_principal():
    print("\n" + "=" * 50)
    print("   SISTEMA DE ANÁLISIS DE CRIMEN URBANO")
    print("=" * 50)
    print("  [CARGA]")
    print("  1. Cargar todos los datasets (raw)")
    print("  2. Cargar un dataset individual (raw)")
    print("  3. Cargar datasets transformados (procesados)")
    print("")
    print("  [ANÁLISIS]")
    print("  4. Analizar datasets")
    print("  5. Comparar columnas entre datasets")
    print("  9. Correlación de Pearson (datasets transformados)")
    print(" 10. Gráficas de hipótesis (H1, H2, H3)")
    print(" 11. Análisis de granularidad (tipo_delito_detalle y descripcion)")
    print(" 12. EDA formal (descriptivas, distribuciones, patrones, peligrosidad)")
    print("")
    print("  [MODELOS]")
    print(" 13. Clustering (geográfico + comportamental por ciudad)")
    print("")
    print("  [FEATURE ENGINEERING]")
    print(" 15. Unificar datasets transformados → crime_unified.csv")
    print(" 16. Generar vector de características → crime_features.csv")
    print(" 17. Cargar crime_unified.csv")
    print(" 18. Cargar crime_features.csv")
    print("")
    print("  [LIMPIEZA]")
    print("  6. Limpiar datasets cargados")
    print("")
    print("  [TRANSFORMACIÓN]")
    print("  7. Transformar datasets limpios")
    print("  8. Exportar datasets transformados a CSV")
    print("")
    print("  0. Salir")
    print("=" * 50)
    return input("  Selecciona una opción: ").strip()


def main():
    datasets_crudos        = {}
    datasets_limpios       = {}
    datasets_transformados = {}
    df_unificado           = None
    df_features            = None

    while True:
        opcion = menu_principal()

        if opcion == "1":
            datasets_crudos = cargar_todos(anio=2025)
            datasets_limpios = {}
            datasets_transformados = {}
            print(f"\n  ✔ Datasets cargados: {list(datasets_crudos.keys())}")

        elif opcion == "2":
            resultado = cargar_individual(anio=2025)
            datasets_crudos.update(resultado)
            datasets_limpios = {}
            datasets_transformados = {}
            if resultado:
                print(f"\n  ✔ Dataset cargado: {list(resultado.keys())[0]}")

        elif opcion == "3":
            datasets_transformados = cargar_transformados()
            datasets_crudos = {}
            datasets_limpios = {}

        elif opcion == "4":
            src = datasets_transformados or datasets_limpios or datasets_crudos
            if _verificar(src, "No hay datasets. Usa opción 1, 2 o 3."):
                load_analyzer.analizar(src)

        elif opcion == "5":
            src = datasets_transformados or datasets_limpios or datasets_crudos
            if _verificar(src, "No hay datasets. Usa opción 1, 2 o 3."):
                column_comparator.comparar(src)

        elif opcion == "6":
            if _verificar(datasets_crudos, "No hay datos crudos. Usa opción 1 o 2 primero."):
                datasets_limpios = cleaner_pipeline.limpiar(datasets_crudos)
                datasets_transformados = {}

        elif opcion == "7":
            if _verificar(datasets_limpios, "No hay datos limpios. Ejecuta la opción 6 primero."):
                datasets_transformados = transformer_pipeline.transformar(datasets_limpios)

        elif opcion == "8":
            if _verificar(datasets_transformados, "No hay datos transformados. Ejecuta la opción 7 primero."):
                exportar(datasets_transformados)

        elif opcion == "9":
            if _verificar(datasets_transformados, "No hay datos transformados. Ejecuta la opción 7 primero."):
                analizar_correlacion(datasets_transformados)

        elif opcion == "10":
            if _verificar(datasets_transformados, "No hay datos transformados. Ejecuta la opción 7 primero."):
                generar_todas(datasets_transformados)

        elif opcion == "11":
            if _verificar(datasets_transformados, "No hay datos transformados. Ejecuta la opción 7 primero."):
                analizar_granularidad(datasets_transformados)

        elif opcion == "12":
            if _verificar(datasets_transformados, "No hay datos transformados. Ejecuta la opción 7 primero."):
                ejecutar_eda(datasets_transformados)

        elif opcion == "13":
            if _verificar(datasets_transformados, "No hay datos transformados. Ejecuta la opción 7 primero."):
                ejecutar_clustering(datasets_transformados)

        elif opcion == "15":
            if _verificar(datasets_transformados, "No hay datos transformados. Ejecuta la opción 7 primero."):
                df_unificado = unificar(datasets_transformados)
                print(f"\n  ✔ Unificado listo: {len(df_unificado):,} filas")

        elif opcion == "16":
            if df_unificado is None:
                print("\n  [AVISO] Primero unifica los datasets (opción 15 o 17).")
            else:
                df_features = generar_features(df_unificado)
                print(f"\n  ✔ Vector listo: {df_features.shape[1]} columnas")

        elif opcion == "17":
            df_unificado = cargar_unificado()

        elif opcion == "18":
            df_features = cargar_features()

        elif opcion == "0":
            print("\n  Hasta luego.\n")
            break

        else:
            print("\n  [ERROR] Opción no válida.")


if __name__ == "__main__":
    main()