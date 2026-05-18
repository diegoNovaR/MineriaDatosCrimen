import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

from src.translations import (
    CIUDADES_LABEL, COLORES_CIUDAD,
    ORDEN_PERIODO, PERIODO_LABEL,
    ORDEN_DIAS, DIA_LABEL, DIAS_ES,
    CATEGORIA_LABEL,
)

RUTA_OUTPUTS = os.path.join("outputs", "eda")
COLS_NUMERICAS = ["hora", "mes", "latitud", "longitud"]


def ejecutar_eda(datasets: dict) -> None:
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EDA — ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 60)

    # ── Por ciudad ──────────────────────────────────────────────
    for ciudad, df in datasets.items():
        _seccion(f"CIUDAD: {CIUDADES_LABEL.get(ciudad, ciudad).upper()}")
        _descriptivas(ciudad, df)
        _boxplots(ciudad, df)

    # ── Distribuciones ──────────────────────────────────────────
    _histogramas(datasets)
    _densidad_hora(datasets)

    # ── Patrones temporales avanzados ───────────────────────────
    _heatmap_hora_mes(datasets)
    _heatmap_hora_diasemana(datasets)

    # ── Peligrosidad ────────────────────────────────────────────
    _peligrosidad_por_periodo(datasets)
    _peligrosidad_por_mes(datasets)
    _peligrosidad_geo(datasets)

    # ── Comparativa entre ciudades ──────────────────────────────
    _comparativa_delitos_mes(datasets)
    _comparativa_peligrosidad(datasets)

    print(f"\n  ✔ EDA completado. Gráficas en: {RUTA_OUTPUTS}/")


# ─── Consola ──────────────────────────────────────────────────────────────────

def _descriptivas(ciudad: str, df: pd.DataFrame) -> None:
    cols = [c for c in COLS_NUMERICAS if c in df.columns]
    if not cols:
        return

    print(f"\n  [ESTADÍSTICAS DESCRIPTIVAS]")
    print(f"  Total registros: {len(df):,}")

    desc = df[cols].describe().T
    fmt  = f"  {'Columna':<12} {'Media':>10} {'Mediana':>10} {'Std':>10} {'Mín':>10} {'Máx':>10}"
    print(fmt)
    print("  " + "-" * 62)
    for col, row in desc.iterrows():
        mediana = df[col].median()
        print(f"  {col:<12} {row['mean']:>10.2f} {mediana:>10.2f} "
              f"{row['std']:>10.2f} {row['min']:>10.2f} {row['max']:>10.2f}")

    # Peligrosidad
    if "es_peligroso" in df.columns:
        n_pel = df["es_peligroso"].sum()
        pct   = n_pel / len(df) * 100
        print(f"\n  Delitos peligrosos : {n_pel:,}  ({pct:.1f}%)")
        print(f"  Delitos no peligrosos: {len(df) - n_pel:,}  ({100 - pct:.1f}%)")

    # Hora más frecuente
    if "hora" in df.columns:
        hora_top = df["hora"].value_counts().idxmax()
        print(f"  Hora más frecuente : {hora_top:02d}:00 h")

    # Mes más frecuente
    if "mes" in df.columns:
        meses_es = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                    7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
        mes_top = df["mes"].value_counts().idxmax()
        print(f"  Mes con más delitos: {meses_es.get(mes_top, mes_top)}")


# ─── Boxplots ─────────────────────────────────────────────────────────────────

def _boxplots(ciudad: str, df: pd.DataFrame) -> None:
    cols = [c for c in COLS_NUMERICAS if c in df.columns]
    if not cols:
        return

    LABELS_ES = {"hora": "Hora", "mes": "Mes", "latitud": "Latitud", "longitud": "Longitud"}

    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    color = COLORES_CIUDAD.get(ciudad, "#888")
    ciudad_label = CIUDADES_LABEL.get(ciudad, ciudad)

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color="black", linewidth=2))

        # Detección atípicos IQR
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr    = q3 - q1
        n_out  = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()

        ax.set_title(f"{LABELS_ES.get(col, col)}\natípicos: {n_out:,}", fontsize=10)
        ax.set_xticks([])

    fig.suptitle(f"Boxplots — {ciudad_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _guardar(fig, f"eda_boxplots_{ciudad}.png")

    # Consola
    print(f"\n  [BOXPLOTS / ATÍPICOS]")
    for col in cols:
        data = df[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr    = q3 - q1
        n_out  = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        print(f"  {LABELS_ES.get(col, col):<12}  Q1={q1:.2f}  Q3={q3:.2f}  "
              f"IQR={iqr:.2f}  atípicos={n_out:,}")


# ─── Histogramas ──────────────────────────────────────────────────────────────

def _histogramas(datasets: dict) -> None:
    cols = ["hora", "mes"]
    LABELS_ES = {"hora": "Hora del día", "mes": "Mes"}

    for col in cols:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
        fig.suptitle(f"Distribución de '{LABELS_ES[col]}' por ciudad",
                     fontsize=13, fontweight="bold")

        for ax, (ciudad, df) in zip(axes, datasets.items()):
            if col not in df.columns:
                continue
            color = COLORES_CIUDAD.get(ciudad, "#888")
            bins  = 24 if col == "hora" else 12
            ax.hist(df[col].dropna(), bins=bins, color=color,
                    edgecolor="white", alpha=0.85)
            ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad))
            ax.set_xlabel(LABELS_ES[col])
            ax.set_ylabel("Frecuencia")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        plt.tight_layout()
        _guardar(fig, f"eda_histograma_{col}.png")


# ─── Densidad por hora ────────────────────────────────────────────────────────

def _densidad_hora(datasets: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    for ciudad, df in datasets.items():
        if "hora" not in df.columns:
            continue
        color = COLORES_CIUDAD.get(ciudad, "#888")
        label = CIUDADES_LABEL.get(ciudad, ciudad)
        sns.kdeplot(df["hora"].dropna(), ax=ax, color=color,
                    label=label, linewidth=2.5, fill=True, alpha=0.15)

    ax.set_title("Densidad de delitos por hora del día — comparativa",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Densidad")
    ax.legend(title="Ciudad")
    plt.tight_layout()
    _guardar(fig, "eda_densidad_hora_comparativa.png")


# ─── Heatmap hora × mes ───────────────────────────────────────────────────────

def _heatmap_hora_mes(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Patrón temporal: Hora × Mes (densidad de delitos)",
                 fontsize=13, fontweight="bold")

    MESES_ES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        if "hora" not in df.columns or "mes" not in df.columns:
            continue

        pivot = (df.groupby(["hora", "mes"])
                   .size()
                   .unstack(fill_value=0))
        pivot.columns = [MESES_ES.get(m, m) for m in pivot.columns]

        sns.heatmap(pivot, ax=ax, cmap="YlOrRd",
                    cbar_kws={"label": "Nº delitos"},
                    linewidths=0.1)
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=11)
        ax.set_xlabel("Mes")
        ax.set_ylabel("Hora")

    plt.tight_layout()
    _guardar(fig, "eda_heatmap_hora_mes.png")


# ─── Heatmap hora × día semana ────────────────────────────────────────────────

def _heatmap_hora_diasemana(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Patrón temporal: Hora × Día de la semana",
                 fontsize=13, fontweight="bold")

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        if "hora" not in df.columns or "dia_semana" not in df.columns:
            continue

        pivot = (df.groupby(["hora", "dia_semana"])
                   .size()
                   .unstack(fill_value=0))

        # Ordenar columnas lun→dom y traducir
        cols_ord = [d for d in ORDEN_DIAS if d in pivot.columns]
        pivot    = pivot[cols_ord]
        pivot.columns = [DIA_LABEL.get(d, d) for d in pivot.columns]

        sns.heatmap(pivot, ax=ax, cmap="YlOrRd",
                    cbar_kws={"label": "Nº delitos"},
                    linewidths=0.1)
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=11)
        ax.set_xlabel("Día")
        ax.set_ylabel("Hora")

    plt.tight_layout()
    _guardar(fig, "eda_heatmap_hora_diasemana.png")


# ─── Peligrosidad × periodo ───────────────────────────────────────────────────

def _peligrosidad_por_periodo(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Delitos peligrosos vs no peligrosos por franja horaria",
                 fontsize=13, fontweight="bold")

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        if "es_peligroso" not in df.columns or "periodo_dia" not in df.columns:
            continue

        pivot = (df.groupby(["periodo_dia", "es_peligroso"])
                   .size()
                   .unstack(fill_value=0)
                   .reindex([p for p in ORDEN_PERIODO if p in df["periodo_dia"].values]))

        pivot.index = [PERIODO_LABEL.get(p, p) for p in pivot.index]
        pivot.columns = ["No peligroso", "Peligroso"]

        pivot.plot(kind="bar", ax=ax, color=["#90CAF9", "#EF5350"],
                   edgecolor="white", width=0.7)
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=11)
        ax.set_xlabel("Franja horaria")
        ax.set_ylabel("Nº delitos")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=8)

    plt.tight_layout()
    _guardar(fig, "eda_peligrosidad_periodo.png")


# ─── Peligrosidad × mes ───────────────────────────────────────────────────────

def _peligrosidad_por_mes(datasets: dict) -> None:
    MESES_ES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Evolución mensual: delitos peligrosos vs no peligrosos",
                 fontsize=13, fontweight="bold")

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        if "es_peligroso" not in df.columns or "mes" not in df.columns:
            continue

        pivot = (df.groupby(["mes", "es_peligroso"])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(range(1, 13), fill_value=0))
        pivot.index = [MESES_ES[m] for m in pivot.index]
        pivot.columns = ["No peligroso", "Peligroso"]

        pivot.plot(kind="line", ax=ax, marker="o",
                   color=["#42A5F5", "#EF5350"], linewidth=2)
        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=11)
        ax.set_xlabel("Mes")
        ax.set_ylabel("Nº delitos")
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=8)

    plt.tight_layout()
    _guardar(fig, "eda_peligrosidad_mes.png")


# ─── Peligrosidad geográfica ──────────────────────────────────────────────────

def _peligrosidad_geo(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Distribución geográfica: peligrosos (rojo) vs no peligrosos (azul)",
                 fontsize=13, fontweight="bold")

    for ax, (ciudad, df) in zip(axes, datasets.items()):
        if not all(c in df.columns for c in ["latitud", "longitud", "es_peligroso"]):
            continue

        no_pel = df[df["es_peligroso"] == False].dropna(subset=["latitud", "longitud"])
        pel    = df[df["es_peligroso"] == True].dropna(subset=["latitud", "longitud"])

        ax.scatter(no_pel["longitud"], no_pel["latitud"],
                   s=0.3, alpha=0.3, color="#42A5F5", label="No peligroso")
        ax.scatter(pel["longitud"], pel["latitud"],
                   s=0.3, alpha=0.4, color="#EF5350", label="Peligroso")

        ax.set_title(CIUDADES_LABEL.get(ciudad, ciudad), fontsize=11)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.legend(markerscale=8, fontsize=8)

    plt.tight_layout()
    _guardar(fig, "eda_peligrosidad_geo.png")


# ─── Comparativa delitos × mes ────────────────────────────────────────────────

def _comparativa_delitos_mes(datasets: dict) -> None:
    MESES_ES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

    fig, ax = plt.subplots(figsize=(13, 5))

    for ciudad, df in datasets.items():
        if "mes" not in df.columns:
            continue
        conteo = df.groupby("mes").size().reindex(range(1, 13), fill_value=0)
        ax.plot(
            [MESES_ES[m] for m in conteo.index],
            conteo.values,
            marker="o", linewidth=2.5,
            label=CIUDADES_LABEL.get(ciudad, ciudad),
            color=COLORES_CIUDAD.get(ciudad, "#888"),
        )

    ax.set_title("Comparativa: volumen de delitos por mes entre ciudades",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Nº delitos")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Ciudad")
    plt.tight_layout()
    _guardar(fig, "eda_comparativa_delitos_mes.png")


# ─── Comparativa peligrosidad entre ciudades ──────────────────────────────────

def _comparativa_peligrosidad(datasets: dict) -> None:
    registros = []
    for ciudad, df in datasets.items():
        if "es_peligroso" not in df.columns:
            continue
        total  = len(df)
        pel    = df["es_peligroso"].sum()
        no_pel = total - pel
        registros.append({
            "ciudad":        CIUDADES_LABEL.get(ciudad, ciudad),
            "Peligroso":     pel / total * 100,
            "No peligroso":  no_pel / total * 100,
        })

    if not registros:
        return

    tabla = pd.DataFrame(registros).set_index("ciudad")

    fig, ax = plt.subplots(figsize=(9, 5))
    tabla.plot(kind="bar", ax=ax, color=["#EF5350", "#42A5F5"],
               edgecolor="white", width=0.6)
    ax.set_title("Comparativa: % delitos peligrosos vs no peligrosos por ciudad",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Ciudad")
    ax.set_ylabel("% del total de delitos")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(title="Tipo")

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _guardar(fig, "eda_comparativa_peligrosidad.png")


# ─── Helper ───────────────────────────────────────────────────────────────────

def _guardar(fig: plt.Figure, nombre: str) -> None:
    ruta = os.path.join(RUTA_OUTPUTS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gráfica] Guardada → {ruta}")


def _seccion(titulo: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")