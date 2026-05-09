import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RUTA_OUTPUTS = os.path.join("outputs")

COLS_NUMERICAS = [
    "anio", "mes", "hora",
    "es_fin_semana", "es_peligroso",
    "latitud", "longitud",
]


def analizar_correlacion(datasets: dict) -> None:
    os.makedirs(RUTA_OUTPUTS, exist_ok=True)

    for ciudad, df in datasets.items():
        print(f"\n{'='*50}")
        print(f"  CORRELACIÓN DE PEARSON: {ciudad.replace('_', ' ').upper()}")
        print(f"{'='*50}")

        cols = [c for c in COLS_NUMERICAS if c in df.columns]
        df_num = df[cols].copy()

        # Convertir booleanos a int para Pearson
        for col in ["es_fin_semana", "es_peligroso"]:
            if col in df_num.columns:
                df_num[col] = df_num[col].astype(int)

        correlacion = df_num.corr(method="pearson")

        # Imprimir por consola
        print(f"\n  Columnas incluidas: {cols}")
        print(f"\n{correlacion.round(3).to_string()}")

        # Pares con correlación relevante (|r| > 0.1, excluyendo diagonal)
        print(f"\n  Pares con |r| > 0.1:")
        hay = False
        for i in range(len(correlacion.columns)):
            for j in range(i + 1, len(correlacion.columns)):
                r = correlacion.iloc[i, j]
                if abs(r) > 0.1:
                    c1 = correlacion.columns[i]
                    c2 = correlacion.columns[j]
                    print(f"    {c1} ↔ {c2}: {r:.3f}")
                    hay = True
        if not hay:
            print("    (ninguno supera el umbral)")

        _graficar(ciudad, correlacion)


def _graficar(ciudad: str, correlacion: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        correlacion,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(
        f"Correlación de Pearson — {ciudad.replace('_', ' ').title()}",
        fontsize=13,
        pad=12,
    )
    plt.tight_layout()

    ruta = os.path.join(RUTA_OUTPUTS, f"correlacion_{ciudad}.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"\n  [gráfica] Guardada en: {ruta}")