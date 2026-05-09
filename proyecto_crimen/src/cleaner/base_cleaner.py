from abc import ABC, abstractmethod
import pandas as pd


class BaseCleaner(ABC):
    """Clase base abstracta para limpieza de datasets."""

    CIUDAD = ""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def limpiar(self) -> pd.DataFrame:
        """Pipeline completo de limpieza."""
        print(f"\n{'='*50}")
        print(f"  LIMPIEZA: {self.CIUDAD}")
        print(f"{'='*50}")
        print(f"  Filas iniciales: {len(self.df):,}")

        self.df = self._convertir_tipos()
        self.df = self._eliminar_columnas()
        self.df = self._tratar_nulos()
        self.df = self._eliminar_outliers_geo()
        self.df = self._texto_a_minusculas()

        print(f"  Filas finales  : {len(self.df):,}")
        return self.df

    @abstractmethod
    def _convertir_tipos(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _eliminar_columnas(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _tratar_nulos(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _eliminar_outliers_geo(self) -> pd.DataFrame:
        pass

    def _texto_a_minusculas(self) -> pd.DataFrame:
        """Convierte todas las columnas object a minúsculas."""
        cols_obj = self.df.select_dtypes(include="object").columns
        for col in cols_obj:
            self.df[col] = self.df[col].str.lower().str.strip()
        print(f"  [texto] Columnas en minúsculas: {list(cols_obj)}")
        return self.df

    def _drop_outliers_geo(self, df: pd.DataFrame, bbox: tuple) -> pd.DataFrame:
        """Elimina filas fuera del bounding box."""
        lat_min, lat_max, lon_min, lon_max = bbox
        antes = len(df)
        mask = (
            df["latitud"].notna() & df["longitud"].notna() &
            df["latitud"].between(lat_min, lat_max) &
            df["longitud"].between(lon_min, lon_max)
        )
        df = df[mask].copy()
        print(f"  [geo] Outliers eliminados: {antes - len(df):,}")
        return df