import pandas as pd
from src.cleaner.base_cleaner import BaseCleaner

BBOX = (41.60, 42.05, -87.95, -87.50)


class ChicagoCleaner(BaseCleaner):

    CIUDAD = "Chicago"

    COLS_ELIMINAR = [
        "ubicacion",        # redundante con latitud/longitud
        "actualizado_en",   # no relevante para análisis
        "coordenada_x",     # redundante con longitud
        "coordenada_y",     # redundante con latitud
    ]

    def _convertir_tipos(self) -> pd.DataFrame:
        df = self.df

        df["fecha"] = pd.to_datetime(df["fecha"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        df["actualizado_en"] = pd.to_datetime(df["actualizado_en"], errors="coerce")

        df["arresto"] = df["arresto"].astype(bool)
        df["violencia_domestica"] = df["violencia_domestica"].astype(bool)

        for col in ["municipio", "area_comunitaria", "distrito", "sector"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        print(f"  [tipos] Fechas, booleanos y enteros convertidos.")
        return df

    def _eliminar_columnas(self) -> pd.DataFrame:
        cols = [c for c in self.COLS_ELIMINAR if c in self.df.columns]
        self.df = self.df.drop(columns=cols)
        print(f"  [cols] Eliminadas: {cols}")
        return self.df

    def _tratar_nulos(self) -> pd.DataFrame:
        antes = len(self.df)
        self.df = self.df.dropna()
        print(f"  [nulos] Filas eliminadas por nulos: {antes - len(self.df):,}")
        return self.df

    def _eliminar_outliers_geo(self) -> pd.DataFrame:
        return self._drop_outliers_geo(self.df, BBOX)