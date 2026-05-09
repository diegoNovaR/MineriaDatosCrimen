import pandas as pd
from src.transformer.base_transformer import BaseTransformer, MAPEO_CHICAGO, COLUMNAS_FINALES


class ChicagoTransformer(BaseTransformer):

    CIUDAD = "Chicago"

    def _extraer_temporales(self) -> pd.DataFrame:
        # Chicago: fecha ya es datetime, no tiene columna hora separada
        return self._calcular_temporales(self.df, col_fecha="fecha")

    def _unificar_tipo_delito(self) -> pd.DataFrame:
        self.df["categoria_delito"] = (
            self.df["tipo_delito"]
            .map(MAPEO_CHICAGO)
            .fillna("otros")
        )
        n_otros = (self.df["categoria_delito"] == "otros").sum()
        print(f"  [categoria] Mapeados. Sin categoría ('otros'): {n_otros:,}")
        return self.df

    def _seleccionar_columnas(self) -> pd.DataFrame:
        self.df["ciudad"] = "chicago"
        cols = [c for c in COLUMNAS_FINALES if c in self.df.columns]
        return self.df[cols]