import pandas as pd
from src.transformer.chicago_transformer import ChicagoTransformer
from src.transformer.phila_transformer import PhilaTransformer
from src.transformer.sf_transformer import SFTransformer

TRANSFORMERS = {
    "chicago":       ChicagoTransformer,
    "philadelphia":  PhilaTransformer,
    "san_francisco": SFTransformer,
}


def transformar(datasets: dict) -> dict:
    """
    Recibe dict ciudad -> DataFrame limpio.
    Retorna dict ciudad -> DataFrame transformado.
    """
    datasets_transformados = {}
    for ciudad, df in datasets.items():
        if ciudad not in TRANSFORMERS:
            print(f"\n[AVISO] No hay transformer definido para: {ciudad}")
            datasets_transformados[ciudad] = df
            continue
        transformer = TRANSFORMERS[ciudad](df)
        datasets_transformados[ciudad] = transformer.transformar()

    print(f"\n  ✔ Transformación completada: {list(datasets_transformados.keys())}")
    return datasets_transformados