import pandas as pd
from src.cleaner.chicago_cleaner import ChicagoCleaner
from src.cleaner.phila_cleaner import PhilaCleaner
from src.cleaner.sf_cleaner import SFCleaner

CLEANERS = {
    "chicago":       ChicagoCleaner,
    "philadelphia":  PhilaCleaner,
    "san_francisco": SFCleaner,
}


def limpiar(datasets: dict) -> dict:
    """
    Recibe dict ciudad -> DataFrame crudo.
    Retorna dict ciudad -> DataFrame limpio.
    """
    datasets_limpios = {}
    for ciudad, df in datasets.items():
        if ciudad not in CLEANERS:
            print(f"\n[AVISO] No hay cleaner definido para: {ciudad}")
            datasets_limpios[ciudad] = df
            continue
        cleaner = CLEANERS[ciudad](df)
        datasets_limpios[ciudad] = cleaner.limpiar()

    print(f"\n  ✔ Limpieza completada: {list(datasets_limpios.keys())}")
    return datasets_limpios