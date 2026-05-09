import pandas as pd
from src.cleaner.base_cleaner import BaseCleaner

BBOX = (37.65, 37.95, -122.55, -122.33)

# Columnas object que van como "no registrado" si son nulas
COLS_NO_REGISTRADO = [
    "tipo_delito", "subtipo_delito", "interseccion",
    "vecindario", "ubicacion",
]

# Columnas donde nulos implican eliminar la fila
COLS_DROP_NULOS = [
    "latitud", "longitud", "tipo_delito",
    "subtipo_delito", "id_cnn", "distrito_supervisor",
]


class SFCleaner(BaseCleaner):

    CIUDAD = "San Francisco"

    COLS_ELIMINAR = [
        "numero_cad",
        "registrado_online",
        "dato_hasta",
        "cargado_en",
        "distrito_supervisor_2012",  # redundante con distrito_supervisor
        "ubicacion",                 # redundante con latitud/longitud
    ]

    def _convertir_tipos(self) -> pd.DataFrame:
        df = self.df

        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["fecha_reporte"] = pd.to_datetime(df["fecha_reporte"], errors="coerce")

        for col in ["distrito_supervisor", "codigo_delito", "id_cnn"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        print(f"  [tipos] Fechas y enteros convertidos.")
        return df

    def _eliminar_columnas(self) -> pd.DataFrame:
        cols = [c for c in self.COLS_ELIMINAR if c in self.df.columns]
        self.df = self.df.drop(columns=cols)
        print(f"  [cols] Eliminadas: {cols}")
        return self.df

    def _tratar_nulos(self) -> pd.DataFrame:
        # 1. Rellenar object con "no registrado"
        for col in COLS_NO_REGISTRADO:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("no registrado")

        # 2. Eliminar filas nulas en columnas críticas
        cols_criticas = [c for c in COLS_DROP_NULOS if c in self.df.columns]
        antes = len(self.df)
        self.df = self.df.dropna(subset=cols_criticas)
        print(f"  [nulos] Filas eliminadas por nulos críticos: {antes - len(self.df):,}")
        return self.df

    def _eliminar_outliers_geo(self) -> pd.DataFrame:
        return self._drop_outliers_geo(self.df, BBOX)