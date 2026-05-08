import pandas as pd
from src.loader.base_loader import BaseLoader


class PhilaLoader(BaseLoader):

    CIUDAD = "Philadelphia"

    COLUMNAS_ES = {
        "the_geom": "geometria",
        "cartodb_id": "id_cartodb",
        "the_geom_webmercator": "geometria_webmercator",
        "objectid": "id",
        "dc_dist": "distrito",
        "psa": "area_servicio_policial",
        "dispatch_date_time": "fecha_despacho",
        "dispatch_date": "fecha",
        "dispatch_time": "hora_despacho",
        "hour": "hora",
        "dc_key": "clave_caso",
        "location_block": "bloque",
        "ucr_general": "codigo_ucr",
        "text_general_code": "tipo_delito",
        "point_x": "longitud",
        "point_y": "latitud",
        "lat": "latitud_alt",
        "lng": "longitud_alt",
    }

    COLUMNA_ANIO = "anio"

    def _leer_csv(self) -> pd.DataFrame:
        return pd.read_csv(
            self.ruta_archivo,
            low_memory=False,
        )

    def _limpiar_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=self.COLUMNAS_ES)

        # Extraer año desde fecha_despacho o fecha
        if "fecha_despacho" in df.columns:
            df["anio"] = pd.to_datetime(
                df["fecha_despacho"], errors="coerce"
            ).dt.year
        elif "fecha" in df.columns:
            df["anio"] = pd.to_datetime(
                df["fecha"], errors="coerce"
            ).dt.year

        # Usar latitud/longitud principales (point_y / point_x)
        # y descartar duplicados alternativos
        if "latitud_alt" in df.columns and "longitud_alt" in df.columns:
            df = df.drop(columns=["latitud_alt", "longitud_alt"])

        return df