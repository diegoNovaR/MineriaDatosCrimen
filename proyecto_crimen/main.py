import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.loader.chicago_loader import ChicagoLoader
from src.loader.phila_loader import PhilaLoader
from src.loader.sf_loader import SFLoader
from src.analyzer import load_analyzer, column_comparator
from src.cleaner import cleaner_pipeline

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


# ─── Carga ────────────────────────────────────────────────────────────────────

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

def _verificar_datasets(datasets: dict, nombre: str = "crudos") -> bool:
    if not datasets:
        print(f"\n  [AVISO] No hay datasets {nombre}. Carga primero (opción 1 o 2).")
        return False
    return True


# ─── Menú ─────────────────────────────────────────────────────────────────────

def menu_principal():
    print("\n" + "=" * 50)
    print("   SISTEMA DE ANÁLISIS DE CRIMEN URBANO")
    print("=" * 50)
    print("  [CARGA]")
    print("  1. Cargar todos los datasets")
    print("  2. Cargar un dataset individual")
    print("")
    print("  [ANÁLISIS]")
    print("  3. Analizar datasets cargados")
    print("  4. Comparar columnas entre datasets")
    print("")
    print("  [LIMPIEZA]")
    print("  5. Limpiar datasets cargados")
    print("")
    print("  0. Salir")
    print("=" * 50)
    return input("  Selecciona una opción: ").strip()


def main():
    datasets_crudos = {}
    datasets_limpios = {}

    while True:
        opcion = menu_principal()

        if opcion == "1":
            datasets_crudos = cargar_todos(anio=2025)
            datasets_limpios = {}
            print(f"\n  ✔ Datasets cargados: {list(datasets_crudos.keys())}")

        elif opcion == "2":
            resultado = cargar_individual(anio=2025)
            datasets_crudos.update(resultado)
            datasets_limpios = {}
            if resultado:
                print(f"\n  ✔ Dataset cargado: {list(resultado.keys())[0]}")

        elif opcion == "3":
            src = datasets_limpios if datasets_limpios else datasets_crudos
            if _verificar_datasets(src):
                load_analyzer.analizar(src)

        elif opcion == "4":
            src = datasets_limpios if datasets_limpios else datasets_crudos
            if _verificar_datasets(src):
                column_comparator.comparar(src)

        elif opcion == "5":
            if _verificar_datasets(datasets_crudos):
                datasets_limpios = cleaner_pipeline.limpiar(datasets_crudos)

        elif opcion == "0":
            print("\n  Hasta luego.\n")
            break

        else:
            print("\n  [ERROR] Opción no válida.")


if __name__ == "__main__":
    main()