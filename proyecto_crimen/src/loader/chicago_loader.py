import pandas as pd
from src.loader.base_loader import BaseLoader


class ChicagoLoader(BaseLoader):

    CIUDAD = "Chicago"

    COLUMNAS_ES = {
        "ID": "id",
        "Case Number": "numero_caso",
        "Date": "fecha",
        "Block": "bloque",
        "IUCR": "codigo_iucr",
        "Primary Type": "tipo_delito",
        "Description": "descripcion",
        "Location Description": "descripcion_lugar",
        "Arrest": "arresto",
        "Domestic": "violencia_domestica",
        "Beat": "sector",
        "District": "distrito",
        "Ward": "municipio",
        "Community Area": "area_comunitaria",
        "FBI Code": "codigo_fbi",
        "X Coordinate": "coordenada_x",
        "Y Coordinate": "coordenada_y",
        "Year": "anio",
        "Updated On": "actualizado_en",
        "Latitude": "latitud",
        "Longitude": "longitud",
        "Location": "ubicacion",
    }

    COLUMNA_ANIO = "anio"

    def _leer_csv(self) -> pd.DataFrame:
        return pd.read_csv(
            self.ruta_archivo,
            low_memory=False,
        )

    def _limpiar_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=self.COLUMNAS_ES)

        # Normalizar decimales: coma → punto en latitud/longitud
        for col in ["latitud", "longitud"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df