import pandas as pd
from src.cleaner.base_cleaner import BaseCleaner

BBOX = (39.85, 40.15, -75.30, -74.95)


class PhilaCleaner(BaseCleaner):

    CIUDAD = "Philadelphia"

    COLS_ELIMINAR = [
        "geometria",
        "geometria_webmercator",
        "id_cartodb",           # redundante con id
        "hora_despacho",        # redundante con hora
    ]

    def _convertir_tipos(self) -> pd.DataFrame:
        df = self.df

        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["fecha_despacho"] = pd.to_datetime(df["fecha_despacho"], errors="coerce")

        df["hora"] = pd.to_numeric(df["hora"], errors="coerce").astype("Int64")
        df["distrito"] = pd.to_numeric(df["distrito"], errors="coerce").astype("Int64")
        df["codigo_ucr"] = pd.to_numeric(df["codigo_ucr"], errors="coerce").astype("Int64")

        print(f"  [tipos] Fechas y enteros convertidos.")
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