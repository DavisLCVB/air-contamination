"""Limpieza e imputación de contaminantes del aire (Lima Metropolitana)."""
from __future__ import annotations

import pandas as pd
import numpy as np

# Pin pyarrow<20 evita un segfault de deploy con pandas 3.x (ver pyproject.toml);
# este flag es una capa extra, no el fix real.
pd.set_option("future.infer_string", False)

# --- Constantes -----------------------------------------------------------

CONTAMINANTES = ["pm_10", "pm_25", "so2", "no2", "o3", "co"]
FECHA_CORTE = pd.Timestamp("2014-10-01")  # pm_25 es muy ruidoso/incompleto antes de esto
LIMITE_INTERPOLACION_HORAS = 6  # máx. horas consecutivas rellenadas por interpolación lineal
SEED = 96  # semilla única del proyecto (compartida con models.py y forecast.py)

np.random.seed(SEED)


# --- Carga ------------------------------------------------------------------

def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """Carga el CSV crudo de SENAMHI y normaliza columnas/tipos."""
    df = pd.read_csv(ruta_csv)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )
    df["estacion"] = df["estacion"].str.strip()
    df["fecha_hora"] = pd.to_datetime(
        dict(year=df.ano, month=df.mes, day=df.dia, hour=df.hora)
    )
    return df


# --- Limpieza e imputación ---------------------------------------------------

def limpiar_e_imputar(
    df: pd.DataFrame,
    fecha_corte: pd.Timestamp = FECHA_CORTE,
    limite_interpolacion: int = LIMITE_INTERPOLACION_HORAS,
    contaminantes: list[str] = CONTAMINANTES,
) -> pd.DataFrame:
    """Filtra, deduplica, rellena la grilla horaria e imputa en 3 pasos."""
    df_ventana = df[df["fecha_hora"] >= fecha_corte].copy()
    df_dedup = df_ventana.groupby(
        ["estacion", "codigo_estacion", "fecha_hora"], as_index=False
    )[contaminantes].mean()

    # grilla horaria completa por estación -> huecos explícitos como NaN
    frames = []
    for est, g in df_dedup.groupby("estacion"):
        idx = pd.date_range(g["fecha_hora"].min(), g["fecha_hora"].max(), freq="h")
        g2 = g.set_index("fecha_hora").reindex(idx)
        g2["estacion"] = est
        g2.index.name = "fecha_hora"
        frames.append(g2)
    df_full = pd.concat(frames).reset_index()
    df_full["mes"] = df_full["fecha_hora"].dt.month
    df_full["hora"] = df_full["fecha_hora"].dt.hour
    df_full["anio"] = df_full["fecha_hora"].dt.year

    for col in contaminantes:
        df_full[f"{col}_imputado"] = df_full[col].isna()

    # 1. interpolación lineal intra-estación (huecos cortos)
    for col in contaminantes:
        df_full[col] = df_full.groupby("estacion")[col].transform(
            lambda s: s.interpolate(
                method="linear", limit=limite_interpolacion, limit_direction="both"
            )
        )

    # 2. climatología fina: mediana por (estación, mes, hora)
    for col in contaminantes:
        perfil_fino = df_full.groupby(["estacion", "mes", "hora"])[col].transform("median")
        df_full[col] = df_full[col].fillna(perfil_fino)

    # 3. climatología gruesa: mediana por (estación, hora) -- fallback
    for col in contaminantes:
        perfil_grueso = df_full.groupby(["estacion", "hora"])[col].transform("median")
        df_full[col] = df_full[col].fillna(perfil_grueso)

    nulos_restantes = df_full[contaminantes].isna().sum()
    if nulos_restantes.any():
        raise ValueError(
            "Quedaron NaN sin imputar tras los 3 pasos de imputación:\n"
            f"{nulos_restantes[nulos_restantes > 0]}"
        )

    return df_full


def cargar_y_limpiar(ruta_csv: str, **kwargs) -> pd.DataFrame:
    """`cargar_datos` + `limpiar_e_imputar` en una sola llamada."""
    df = cargar_datos(ruta_csv)
    return limpiar_e_imputar(df, **kwargs)


def guardar_parquet(df: pd.DataFrame, ruta_salida: str) -> None:
    """Persiste el DataFrame limpio en parquet (preserva dtypes)."""
    df.to_parquet(ruta_salida, index=False)


def reporte_trazabilidad(df: pd.DataFrame, columna: str = "pm_25") -> pd.Series:
    """% de filas imputadas por estación para `columna`."""
    col_imputado = f"{columna}_imputado"
    return df.groupby("estacion")[col_imputado].mean() * 100


if __name__ == "__main__":
    RUTA_DATOS = "./data/air_contamination.csv"
    RUTA_LIMPIO = "./data/air_contamination_clean.parquet"

    df_limpio = cargar_y_limpiar(RUTA_DATOS)
    guardar_parquet(df_limpio, RUTA_LIMPIO)

    print("Shape final:", df_limpio.shape)
    print("\n% de NaN por columna de contaminante:")
    print((df_limpio[CONTAMINANTES].isna().mean() * 100).round(2))
    print("\nTrazabilidad pm_25 (% de filas imputadas por estación):")
    print(reporte_trazabilidad(df_limpio, "pm_25").round(2))
