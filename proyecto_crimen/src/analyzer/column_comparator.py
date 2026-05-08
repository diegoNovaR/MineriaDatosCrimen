import pandas as pd
from itertools import combinations


# Grupos de columnas semánticamente equivalentes entre ciudades
GRUPOS_SEMANTICOS = [
    ["id", "id_cartodb", "id_incidente"],
    ["fecha", "fecha_incidente", "fecha_despacho"],
    ["tipo_delito", "texto_codigo_general"],
    ["distrito"],
    ["latitud", "latitud"],
    ["longitud", "longitud"],
    ["descripcion"],
    ["bloque"],
    ["anio"],
    ["hora"],
]


def comparar(datasets: dict) -> None:
    """Compara columnas entre todos los datasets cargados."""
    ciudades = list(datasets.keys())
    columnas_por_ciudad = {c: set(datasets[c].columns) for c in ciudades}

    _encabezado()
    _columnas_por_ciudad(columnas_por_ciudad)
    _columnas_comunes(ciudades, columnas_por_ciudad)
    _columnas_unicas(ciudades, columnas_por_ciudad)
    _grupos_equivalentes(ciudades, columnas_por_ciudad)
    _resumen_cobertura(ciudades, columnas_por_ciudad)


# ─── Secciones ────────────────────────────────────────────────────────────────

def _encabezado() -> None:
    print("\n" + "=" * 60)
    print("  COMPARADOR DE COLUMNAS ENTRE DATASETS")
    print("=" * 60)


def _columnas_por_ciudad(columnas_por_ciudad: dict) -> None:
    print(f"\n[COLUMNAS POR CIUDAD]")
    for ciudad, cols in columnas_por_ciudad.items():
        print(f"\n  {ciudad.replace('_', ' ').upper()} ({len(cols)} columnas):")
        for col in sorted(cols):
            print(f"    - {col}")


def _columnas_comunes(ciudades: list, columnas_por_ciudad: dict) -> None:
    print(f"\n[COLUMNAS COMUNES (presentes en todas las ciudades)]")
    todas = [columnas_por_ciudad[c] for c in ciudades]
    comunes = set.intersection(*todas)
    if comunes:
        for col in sorted(comunes):
            print(f"  ✔ {col}")
    else:
        print("  (ninguna columna comparte exactamente el mismo nombre)")

    # Pares de ciudades
    print(f"\n[COLUMNAS COMUNES POR PAR DE CIUDADES]")
    for c1, c2 in combinations(ciudades, 2):
        comunes_par = columnas_por_ciudad[c1] & columnas_por_ciudad[c2]
        label = f"{c1.replace('_',' ').title()} ↔ {c2.replace('_',' ').title()}"
        print(f"\n  {label}: {len(comunes_par)} columnas")
        for col in sorted(comunes_par):
            print(f"    - {col}")


def _columnas_unicas(ciudades: list, columnas_por_ciudad: dict) -> None:
    print(f"\n[COLUMNAS EXCLUSIVAS POR CIUDAD]")
    todas = set.union(*[columnas_por_ciudad[c] for c in ciudades])
    for ciudad in ciudades:
        resto = set.union(*[columnas_por_ciudad[c] for c in ciudades if c != ciudad])
        unicas = columnas_por_ciudad[ciudad] - resto
        print(f"\n  {ciudad.replace('_', ' ').upper()} ({len(unicas)} exclusivas):")
        for col in sorted(unicas):
            print(f"    ~ {col}")


def _grupos_equivalentes(ciudades: list, columnas_por_ciudad: dict) -> None:
    print(f"\n[COLUMNAS SEMÁNTICAMENTE EQUIVALENTES]")
    for grupo in GRUPOS_SEMANTICOS:
        presentes = {}
        for ciudad in ciudades:
            cols_ciudad = columnas_por_ciudad[ciudad]
            encontradas = [c for c in grupo if c in cols_ciudad]
            if encontradas:
                presentes[ciudad] = encontradas

        if len(presentes) > 1:
            resumen = " | ".join(
                f"{c.replace('_',' ').title()}: {', '.join(v)}"
                for c, v in presentes.items()
            )
            print(f"  [{' / '.join(grupo[:2])}...]  →  {resumen}")


def _resumen_cobertura(ciudades: list, columnas_por_ciudad: dict) -> None:
    print(f"\n[RESUMEN DE COBERTURA]")
    todas = set.union(*[columnas_por_ciudad[c] for c in ciudades])
    print(f"  Total columnas distintas entre todos los datasets : {len(todas)}")
    for ciudad in ciudades:
        n = len(columnas_por_ciudad[ciudad])
        print(f"  {ciudad.replace('_',' ').title():<25} {n} columnas")