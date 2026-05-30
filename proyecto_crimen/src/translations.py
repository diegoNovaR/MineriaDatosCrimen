# ─────────────────────────────────────────────────────────────────────────────
# translations.py
# Traducciones centralizadas para todas las visualizaciones y análisis.
# Cualquier cambio aquí se refleja automáticamente en hypothesis_viz.py,
# granularity_analyzer.py y cualquier módulo que lo importe.
# ─────────────────────────────────────────────────────────────────────────────

# ─── Ciudades ─────────────────────────────────────────────────────────────────
CIUDADES_LABEL = {
    "chicago":       "Chicago",
    "philadelphia":  "Philadelphia",
    "san_francisco": "San Francisco",
}

COLORES_CIUDAD = {
    "chicago":       "#2196F3",
    "philadelphia":  "#F44336",
    "san_francisco": "#4CAF50",
}

# ─── Periodos del día ─────────────────────────────────────────────────────────
ORDEN_PERIODO = ["madrugada", "mañana", "tarde", "noche"]

PERIODO_LABEL = {
    "madrugada": "Madrugada\n(0h–5h)",
    "mañana":    "Mañana\n(6h–11h)",
    "tarde":     "Tarde\n(12h–17h)",
    "noche":     "Noche\n(18h–23h)",
}

# ─── Días de la semana ────────────────────────────────────────────────────────
ORDEN_DIAS = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday"
]

DIA_LABEL = {
    "monday":    "Lunes",
    "tuesday":   "Martes",
    "wednesday": "Miércoles",
    "thursday":  "Jueves",
    "friday":    "Viernes",
    "saturday":  "Sábado",
    "sunday":    "Domingo",
}

DIAS_ES = [DIA_LABEL[d] for d in ORDEN_DIAS]

# ─── Categorías de delito (nivel general) ─────────────────────────────────────
CATEGORIA_LABEL = {
    "robo_simple":           "Robo simple",
    "agresion":              "Agresión",
    "daño_propiedad":        "Daño a propiedad",
    "robo_vehiculo":         "Robo de vehículo",
    "fraude_engaño":         "Fraude / Engaño",
    "robo_con_violencia":    "Robo con violencia",
    "allanamiento":          "Allanamiento",
    "drogas":                "Drogas",
    "violacion_armas":       "Violación de armas",
    "agresion_sexual":       "Agresión sexual",
    "delito_contra_menores": "Delito contra menores",
    "homicidio":             "Homicidio",
    "arson":                 "Incendio provocado",
    "amenaza_acoso":         "Amenaza / Acoso / Secuestro",
    "trata_personas":        "Trata de personas",
    "orden_publico":         "Orden público",
    "sin_relevancia":        "Sin clasificación",
    "otros":                 "Otros",
}

# ─── Tipo de delito detalle (nivel medio) ─────────────────────────────────────
TIPO_DETALLE_LABEL = {
    # Chicago
    "theft":                          "Robo",
    "battery":                        "Agresión física",
    "criminal damage":                "Daño criminal",
    "assault":                        "Asalto",
    "motor vehicle theft":            "Robo de vehículo",
    "other offense":                  "Otra infracción",
    "deceptive practice":             "Práctica engañosa",
    "burglary":                       "Allanamiento",
    "narcotics":                      "Narcóticos",
    "robbery":                        "Robo con violencia",
    # Philadelphia
    "thefts":                         "Robos",
    "other assaults":                 "Otras agresiones",
    "all other offenses":             "Otras infracciones",
    "vandalism/criminal mischief":    "Vandalismo",
    "theft from vehicle":             "Robo interior vehículo",
    "fraud":                          "Fraude",
    "aggravated assault no firearm":  "Agresión agravada sin arma",
    "burglary residential":           "Allanamiento residencial",
    "narcotic / drug law violations": "Violaciones ley de drogas",
    # Chicago (adicionales)
    "weapons violation":              "Violación de armas",
    "criminal trespass":              "Allanamiento de morada",
    "criminal sexual assault":        "Agresión sexual",
    "offense involving children":     "Delito contra menores",
    "sex offense":                    "Delito sexual",
    # Philadelphia (adicionales)
    "weapon violations":              "Violaciones de armas",
    "aggravated assault firearm":     "Agresión agravada con arma",
    "robbery no firearm":             "Robo sin arma de fuego",
    "burglary non-residential":       "Allanamiento no residencial",
    "robbery firearm":                "Robo con arma de fuego",
    # San Francisco
    "larceny theft":                  "Hurto",
    "drug offense":                   "Delito de drogas",
    "other miscellaneous":            "Otros misceláneos",
    "malicious mischief":             "Daño malicioso",
    "warrant":                        "Orden judicial",
    "non-criminal":                   "No criminal",
    "lost property":                  "Propiedad perdida",
    "disorderly conduct":             "Conducta desordenada",
    "missing person":                 "Persona desaparecida",
    "suspicious occ":                 "Incidente sospechoso",
    "miscellaneous investigation":    "Investigación miscelánea",
}


def traducir(valor: str, diccionario: dict) -> str:
    """Traduce un valor usando el diccionario dado. Si no existe, retorna el valor original."""
    return diccionario.get(str(valor).lower(), valor) if valor else valor


def traducir_serie(serie, diccionario: dict):
    """Traduce una Serie de pandas usando el diccionario dado."""
    return serie.map(lambda x: diccionario.get(str(x).lower(), x) if x else x)