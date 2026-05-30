from abc import ABC, abstractmethod
import pandas as pd


class BaseTransformer(ABC):
    """Clase base abstracta para transformación de datasets."""

    CIUDAD = ""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def transformar(self) -> pd.DataFrame:
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
        pass

    @abstractmethod
    def _unificar_tipo_delito(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _seleccionar_columnas(self) -> pd.DataFrame:
        pass

    def _agregar_es_peligroso(self) -> pd.DataFrame:
        PELIGROSOS = {
            "agresion",
            "robo_con_violencia",
            "agresion_sexual",
            "delito_contra_menores",
            "violacion_armas",
            "homicidio",
            "arson",
            "amenaza_acoso",
            "trata_personas",
        }
        if "categoria_delito" in self.df.columns:
            self.df["es_peligroso"] = self.df["categoria_delito"].isin(PELIGROSOS)
            print(f"  [peligroso] Peligrosos: {self.df['es_peligroso'].sum():,} / {len(self.df):,}")
        return self.df

    def _calcular_temporales(self, df: pd.DataFrame, col_fecha: str, col_hora: str = None) -> pd.DataFrame:
        PERIODOS = {
            range(0, 6):   "madrugada",
            range(6, 12):  "mañana",
            range(12, 18): "tarde",
            range(18, 24): "noche",
        }

        fecha = df[col_fecha]

        if col_hora and col_hora in df.columns:
            df["hora"] = pd.to_numeric(df[col_hora], errors="coerce").astype("Int64")
        else:
            df["hora"] = fecha.dt.hour.astype("Int64")

        df["mes"]           = fecha.dt.month.astype("Int64")
        df["dia_semana"]    = fecha.dt.day_name().str.lower()
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


# ─── Mapeos ───────────────────────────────────────────────────────────────────

MAPEO_CHICAGO = {
    "theft":                              "robo_simple",
    "battery":                            "agresion",
    "criminal damage":                    "daño_propiedad",
    "assault":                            "agresion",
    "motor vehicle theft":                "robo_vehiculo",
    "other offense":                      "orden_publico",
    "deceptive practice":                 "fraude_engaño",
    "burglary":                           "allanamiento",
    "narcotics":                          "drogas",
    "other narcotic violation":           "drogas",
    "robbery":                            "robo_con_violencia",
    "weapons violation":                  "violacion_armas",
    "concealed carry license violation":  "violacion_armas",
    "criminal trespass":                  "allanamiento",
    "criminal sexual assault":            "agresion_sexual",
    "offense involving children":         "delito_contra_menores",
    "sex offense":                        "agresion_sexual",
    "homicide":                           "homicidio",
    "arson":                              "arson",
    "stalking":                           "amenaza_acoso",
    "intimidation":                       "amenaza_acoso",
    "kidnapping":                         "amenaza_acoso",
    "human trafficking":                  "trata_personas",
    "public peace violation":             "orden_publico",
    "interference with public officer":   "orden_publico",
    "liquor law violation":               "orden_publico",
    "prostitution":                       "orden_publico",
    "gambling":                           "orden_publico",
    "obscenity":                          "orden_publico",
    "public indecency":                   "orden_publico",
    "non-criminal":                       "sin_relevancia",
}

MAPEO_PHILADELPHIA = {
    "thefts":                                   "robo_simple",
    "theft from vehicle":                       "robo_simple",
    "receiving stolen property":                "robo_simple",
    "other assaults":                           "agresion",
    "aggravated assault no firearm":            "agresion",
    "all other offenses":                       "sin_relevancia",
    "motor vehicle theft":                      "robo_vehiculo",
    "vandalism/criminal mischief":              "daño_propiedad",
    "fraud":                                    "fraude_engaño",
    "embezzlement":                             "fraude_engaño",
    "forgery and counterfeiting":               "fraude_engaño",
    "burglary residential":                     "allanamiento",
    "burglary non-residential":                 "allanamiento",
    "narcotic / drug law violations":           "drogas",
    "weapon violations":                        "violacion_armas",
    "aggravated assault firearm":               "robo_con_violencia",
    "robbery no firearm":                       "robo_con_violencia",
    "robbery firearm":                          "robo_con_violencia",
    "rape":                                     "agresion_sexual",
    "other sex offenses (not commercialized)":  "agresion_sexual",
    "offenses against family and children":     "delito_contra_menores",
    "homicide - criminal":                      "homicidio",
    "arson":                                    "arson",
    "disorderly conduct":                       "orden_publico",
    "driving under the influence":              "orden_publico",
    "public drunkenness":                       "orden_publico",
    "vagrancy/loitering":                       "orden_publico",
    "liquor law violations":                    "orden_publico",
    "gambling violations":                      "orden_publico",
    "prostitution and commercialized vice":     "orden_publico",
}

MAPEO_SF = {
    "larceny theft":                                  "robo_simple",
    "stolen property":                                "robo_simple",
    "assault":                                        "agresion",
    "drug offense":                                   "drogas",
    "drug violation":                                 "drogas",
    "other miscellaneous":                            "sin_relevancia",
    "other":                                          "sin_relevancia",
    "other offenses":                                 "sin_relevancia",
    "case closure":                                   "sin_relevancia",
    "courtesy report":                                "sin_relevancia",
    "no registrado":                                  "sin_relevancia",
    "malicious mischief":                             "daño_propiedad",
    "vandalism":                                      "daño_propiedad",
    "warrant":                                        "orden_publico",
    "non-criminal":                                   "sin_relevancia",
    "burglary":                                       "allanamiento",
    "motor vehicle theft":                            "robo_vehiculo",
    "motor vehicle theft?":                           "robo_vehiculo",
    "recovered vehicle":                              "robo_vehiculo",
    "vehicle impounded":                              "robo_vehiculo",
    "vehicle misplaced":                              "robo_vehiculo",
    "fraud":                                          "fraude_engaño",
    "forgery and counterfeiting":                     "fraude_engaño",
    "embezzlement":                                   "fraude_engaño",
    "lost property":                                  "sin_relevancia",
    "missing person":                                 "sin_relevancia",
    "disorderly conduct":                             "orden_publico",
    "civil sidewalks":                                "orden_publico",
    "prostitution":                                   "orden_publico",
    "gambling":                                       "orden_publico",
    "liquor laws":                                    "orden_publico",
    "traffic violation arrest":                       "orden_publico",
    "traffic collision":                              "orden_publico",
    "suspicious occ":                                 "orden_publico",
    "suspicious":                                     "orden_publico",
    "miscellaneous investigation":                    "sin_relevancia",
    "fire report":                                    "sin_relevancia",
    "robbery":                                        "robo_con_violencia",
    "weapons offense":                                "violacion_armas",
    "weapons carrying etc":                           "violacion_armas",
    "weapons offence":                                "violacion_armas",
    "sex offense":                                    "agresion_sexual",
    "rape":                                           "agresion_sexual",
    "offences against the family and children":       "delito_contra_menores",
    "homicide":                                       "homicidio",
    "arson":                                          "arson",
    "suicide":                                        "sin_relevancia",
    "human trafficking (a), commercial sex acts":     "trata_personas",
    "human trafficking, commercial sex acts":         "trata_personas",
}

# ─── Columnas finales ─────────────────────────────────────────────────────────
COLUMNAS_FINALES = [
    "id", "ciudad", "fecha", "anio", "mes", "hora",
    "dia_semana", "es_fin_semana", "periodo_dia",
    "categoria_delito", "es_peligroso",
    "latitud", "longitud",
]