import pandas as pd
from src.transformer.base_transformer import BaseTransformer, MAPEO_PHILADELPHIA, COLUMNAS_FINALES


class PhilaTransformer(BaseTransformer):

    CIUDAD = "Philadelphia"

    def _extraer_temporales(self) -> pd.DataFrame:
        # Philadelphia: tiene columna hora separada como int
        return self._calcular_temporales(self.df, col_fecha="fecha", col_hora="hora")

    def _unificar_tipo_delito(self) -> pd.DataFrame:
        self.df["categoria_delito"] = (
            self.df["tipo_delito"]
            .map(MAPEO_PHILADELPHIA)
            .fillna("otros")
        )
        n_otros = (self.df["categoria_delito"] == "otros").sum()
        print(f"  [categoria] Mapeados. Sin categoría ('otros'): {n_otros:,}")
        return self.df

    def _seleccionar_columnas(self) -> pd.DataFrame:
        self.df["ciudad"] = "philadelphia"
        cols = [c for c in COLUMNAS_FINALES if c in self.df.columns]
        return self.df[cols]