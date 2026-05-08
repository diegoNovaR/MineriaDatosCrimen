from abc import ABC, abstractmethod
import pandas as pd


class BaseLoader(ABC):
    """Clase base abstracta para la carga de datasets de crimen."""

    # Subclases deben definir estos atributos
    CIUDAD = ""
    COLUMNAS_ES = {}       # Mapeo: nombre_original -> nombre_español
    COLUMNA_ANIO = ""      # Nombre de la columna de año (ya traducida)

    def __init__(self, ruta_archivo: str):
        self.ruta_archivo = ruta_archivo
        self.df: pd.DataFrame | None = None

    @abstractmethod
    def _leer_csv(self) -> pd.DataFrame:
        """Lee el CSV con los parámetros específicos de cada ciudad."""
        pass

    @abstractmethod
    def _limpiar_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renombra y ajusta columnas al estándar en español."""
        pass

    def cargar(self, anio: int = 2025) -> pd.DataFrame:
        """
        Pipeline completo de carga:
        1. Lee el CSV
        2. Renombra columnas al español
        3. Filtra por año
        """
        print(f"\n{'='*50}")
        print(f"  Cargando datos: {self.CIUDAD}")
        print(f"{'='*50}")

        df = self._leer_csv()
        print(f"  Filas leídas (total): {len(df):,}")

        df = self._limpiar_columnas(df)

        df = self._filtrar_por_anio(df, anio)
        print(f"  Filas tras filtro ({anio}): {len(df):,}")

        self.df = df
        print(f"  Columnas finales: {list(df.columns)}")
        return df

    def _filtrar_por_anio(self, df: pd.DataFrame, anio: int) -> pd.DataFrame:
        """Filtra el dataframe por año usando COLUMNA_ANIO."""
        if self.COLUMNA_ANIO and self.COLUMNA_ANIO in df.columns:
            df[self.COLUMNA_ANIO] = pd.to_numeric(df[self.COLUMNA_ANIO], errors="coerce")
            return df[df[self.COLUMNA_ANIO] == anio].copy()
        print(f"  [AVISO] No se encontró la columna de año '{self.COLUMNA_ANIO}', no se filtra.")
        return df