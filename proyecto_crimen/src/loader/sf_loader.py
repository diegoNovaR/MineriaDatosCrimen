import pandas as pd
from src.loader.base_loader import BaseLoader


class SFLoader(BaseLoader):

    CIUDAD = "San Francisco"

    COLUMNAS_ES = {
        "Row ID": "id",
        "Incident Datetime": "fecha",
        "Incident Date": "fecha_incidente",
        "Incident Time": "hora",
        "Incident Year": "anio",
        "Incident Day of Week": "dia_semana",
        "Report Datetime": "fecha_reporte",
        "Incident ID": "id_incidente",
        "Incident Number": "numero_incidente",
        "CAD Number": "numero_cad",
        "Report Type Code": "codigo_tipo_reporte",
        "Report Type Description": "descripcion_tipo_reporte",
        "Filed Online": "registrado_online",
        "Incident Code": "codigo_delito",
        "Incident Category": "tipo_delito",
        "Incident Subcategory": "subtipo_delito",
        "Incident Description": "descripcion",
        "Resolution": "resolucion",
        "Intersection": "interseccion",
        "CNN": "id_cnn",
        "Police District": "distrito",
        "Analysis Neighborhood": "vecindario",
        "Supervisor District": "distrito_supervisor",
        "Supervisor District 2012": "distrito_supervisor_2012",
        "Latitude": "latitud",
        "Longitude": "longitud",
        "Point": "ubicacion",
        "data_as_of": "dato_hasta",
        "data_loaded_at": "cargado_en",
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