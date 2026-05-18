from abc import ABC, abstractmethod
import pandas as pd


class BaseTransformer(ABC):
    """Clase base abstracta para transformación de datasets."""

    CIUDAD = ""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def transformar(self) -> pd.DataFrame:
        """Pipeline completo de transformación."""
        print(f"\n{'='*50}")
        print(f"  TRANSFORMACIÓN: {self.CIUDAD}")
        print(f"{'='*50}")
        print(f"  Filas iniciales: {len(self.df):,}")

        self.df = self._extraer_temporales()
        self.df = self._unificar_tipo_delito()
        self.df = self._agregar_es_peligroso()
        self.df = self._seleccionar_columnas()

        print(f"  Filas finales  : {len(self.df):,}")
        print(f"  Columnas finales: {list(self.df.columns)}")
        return self.df

    @abstractmethod
    def _extraer_temporales(self) -> pd.DataFrame:
        """Extrae hora, dia_semana, es_fin_semana, mes, periodo_dia desde fecha."""
        pass

    @abstractmethod
    def _unificar_tipo_delito(self) -> pd.DataFrame:
        """Mapea tipo_delito local a categoría unificada."""
        pass

    @abstractmethod
    def _seleccionar_columnas(self) -> pd.DataFrame:
        """Selecciona y ordena las columnas finales."""
        pass

    def _agregar_es_peligroso(self) -> pd.DataFrame:
        """Flag basado en categoria_delito unificada."""
        PELIGROSOS = {
            "agresion", "robo_con_violencia",
            "agresion_sexual", "delito_contra_menores", "violacion_armas",
        }
        if "categoria_delito" in self.df.columns:
            self.df["es_peligroso"] = self.df["categoria_delito"].isin(PELIGROSOS)
            print(f"  [peligroso] Peligrosos: {self.df['es_peligroso'].sum():,} / {len(self.df):,}")
        return self.df

    def _calcular_temporales(self, df: pd.DataFrame, col_fecha: str, col_hora: str = None) -> pd.DataFrame:
        """
        Calcula features temporales comunes.
        col_hora: nombre de columna hora si ya existe (int), si no se extrae de col_fecha.
        """
        PERIODOS = {
            range(0, 6):   "madrugada",
            range(6, 12):  "mañana",
            range(12, 18): "tarde",
            range(18, 24): "noche",
        }

        fecha = df[col_fecha]

        # Hora
        if col_hora and col_hora in df.columns:
            df["hora"] = pd.to_numeric(df[col_hora], errors="coerce").astype("Int64")
        else:
            df["hora"] = fecha.dt.hour.astype("Int64")

        df["mes"]        = fecha.dt.month.astype("Int64")
        df["dia_semana"] = fecha.dt.day_name().str.lower()
        df["es_fin_semana"] = fecha.dt.dayofweek >= 5

        def get_periodo(h):
            if pd.isna(h):
                return "no registrado"
            for rng, nombre in PERIODOS.items():
                if h in rng:
                    return nombre
            return "no registrado"

        df["periodo_dia"] = df["hora"].apply(get_periodo)

        print(f"  [temporales] hora, mes, dia_semana, es_fin_semana, periodo_dia generados.")
        return df


# ─── Mapeo unificado de categorías ───────────────────────────────────────────

MAPEO_CHICAGO = {
    "theft":                       "robo_simple",
    "battery":                     "agresion",
    "criminal damage":             "daño_propiedad",
    "assault":                     "agresion",
    "motor vehicle theft":         "robo_vehiculo",
    "other offense":               "warrant_otros",
    "deceptive practice":          "fraude_engaño",
    "burglary":                    "allanamiento",
    "narcotics":                   "drogas",
    "robbery":                     "robo_con_violencia",
    "weapons violation":           "violacion_armas",
    "criminal trespass":           "allanamiento",
    "criminal sexual assault":     "agresion_sexual",
    "offense involving children":  "delito_contra_menores",
    "sex offense":                 "agresion_sexual",
}

MAPEO_PHILADELPHIA = {
    "thefts":                         "robo_simple",
    "other assaults":                 "agresion",
    "all other offenses":             "warrant_otros",
    "motor vehicle theft":            "robo_vehiculo",
    "vandalism/criminal mischief":    "daño_propiedad",
    "theft from vehicle":             "robo_simple",
    "fraud":                          "fraude_engaño",
    "aggravated assault no firearm":  "agresion",
    "burglary residential":           "allanamiento",
    "narcotic / drug law violations": "drogas",
    "weapon violations":              "violacion_armas",
    "aggravated assault firearm":     "robo_con_violencia",
    "robbery no firearm":             "robo_con_violencia",
    "burglary non-residential":       "allanamiento",
    "robbery firearm":                "robo_con_violencia",
}

MAPEO_SF = {
    "larceny theft":      "robo_simple",
    "assault":            "agresion",
    "drug offense":       "drogas",
    "other miscellaneous": "warrant_otros",
    "malicious mischief": "daño_propiedad",
    "warrant":            "warrant_otros",
    "burglary":           "allanamiento",
    "motor vehicle theft": "robo_vehiculo",
    "non-criminal":       "warrant_otros",
    "fraud":              "fraude_engaño",
}

# Columnas finales comunes para los 3 datasets
COLUMNAS_FINALES = [
    "id", "ciudad", "fecha", "anio", "mes", "hora",
    "dia_semana", "es_fin_semana", "periodo_dia",
    "categoria_delito",        # nivel general    (ej: robo_simple)
    "tipo_delito_detalle",     # nivel medio       (ej: theft, larceny theft)
    "descripcion",             # nivel fino        (ej: $500 and under) — solo Chicago y SF
    "es_peligroso",
    "latitud", "longitud",
]