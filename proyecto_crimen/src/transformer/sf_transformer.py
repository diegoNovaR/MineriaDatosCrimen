import pandas as pd
from src.transformer.base_transformer import BaseTransformer, MAPEO_SF, COLUMNAS_FINALES


class SFTransformer(BaseTransformer):

    CIUDAD = "San Francisco"

    def _extraer_temporales(self) -> pd.DataFrame:
        # SF: tiene columna hora separada como string "HH:MM"
        if "hora" in self.df.columns:
            self.df["hora"] = (
                self.df["hora"]
                .astype(str)
                .str.split(":")
                .str[0]
            )
        return self._calcular_temporales(self.df, col_fecha="fecha", col_hora="hora")

    def _unificar_tipo_delito(self) -> pd.DataFrame:
        self.df["categoria_delito"] = (
            self.df["tipo_delito"]
            .map(MAPEO_SF)
            .fillna("otros")
        )
        n_otros = (self.df["categoria_delito"] == "otros").sum()
        print(f"  [categoria] Mapeados. Sin categoría ('otros'): {n_otros:,}")
        return self.df

    def _seleccionar_columnas(self) -> pd.DataFrame:
        self.df["ciudad"] = "san_francisco"
        if "tipo_delito" in self.df.columns:
            self.df = self.df.rename(columns={"tipo_delito": "tipo_delito_detalle"})
        cols = [c for c in COLUMNAS_FINALES if c in self.df.columns]
        return self.df[cols]