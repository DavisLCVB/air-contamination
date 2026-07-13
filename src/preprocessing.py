"""Pipeline de limpieza e imputación de contaminantes del aire (Lima Metropolitana).

Refactor 1:1 de las secciones "Configuración", "Carga de Datos" y
"Limpieza e Inputación" de notebooks/01_eda.ipynb. Los parámetros
(FECHA_CORTE, LIMITE_INTERPOLACION_HORAS, SEED) ya fueron justificados
contra el dataset real en ese notebook -- no alterar sin validar de nuevo
ahí primero.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# --------------------------------------------------------------------------
# Constantes finales (notebooks/01_eda.ipynb, celdas 3 y 22)
# --------------------------------------------------------------------------

CONTAMINANTES = ["pm_10", "pm_25", "so2", "no2", "o3", "co"]
"""Columnas de contaminantes horarios presentes en el dataset SENAMHI."""

FECHA_CORTE = pd.Timestamp("2014-10-01")
"""Inicio de la ventana de análisis.

pm_25 no existe antes de 2014 y su % de nulos recién baja de ~50-58% a
22.7% en octubre de 2014 (ver celdas 13-15 del notebook). Antes de esa
fecha la señal es demasiado incompleta para imputar con confianza.
"""

LIMITE_INTERPOLACION_HORAS = 6
"""Máximo de horas consecutivas que se rellenan por interpolación lineal.

La mediana de duración de huecos en pm_25 es 6h (celdas 16-20). Se eligió
ese valor y no 3h (deja más carga a los pasos de climatología) ni 12h
(equivale a medio día de dato inventado por interpolación).
"""

SEED = 96
"""Semilla de aleatoriedad, fijada para reproducibilidad entre notebook y
módulo (celda 3). Compartida con models.py y forecast.py."""

np.random.seed(SEED)


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------


def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """Carga el CSV crudo de SENAMHI y normaliza nombres de columnas y tipos.

    No filtra por fecha ni imputa nada: solo dexa el DataFrame en un estado
    consistente (columnas en snake_case, `estacion` sin espacios sobrantes,
    `fecha_hora` como datetime) para que `limpiar_e_imputar` pueda operar
    sobre él sin sorpresas de formato.

    Parameters
    ----------
    ruta_csv:
        Ruta al CSV crudo (columnas tipo "CODIGO ESTACION", "PM 2.5", etc.).

    Returns
    -------
    DataFrame con columnas normalizadas y `fecha_hora` construida a partir
    de `ano`, `mes`, `dia`, `hora`.
    """
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


# --------------------------------------------------------------------------
# Limpieza e imputación
# --------------------------------------------------------------------------


def limpiar_e_imputar(
    df: pd.DataFrame,
    fecha_corte: pd.Timestamp = FECHA_CORTE,
    limite_interpolacion: int = LIMITE_INTERPOLACION_HORAS,
    contaminantes: list[str] = CONTAMINANTES,
) -> pd.DataFrame:
    """Aplica el pipeline completo de limpieza e imputación.

    Pasos (en este orden, no intercambiable):
    1. Filtra la ventana `fecha_hora >= fecha_corte` -- antes de esa fecha
       pm_25 es demasiado ruidoso para imputar (ver `FECHA_CORTE`).
    2. Deduplica por (estacion, codigo_estacion, fecha_hora) promediando
       los contaminantes -- el dataset trae registros duplicados por
       timestamp/estación.
    3. Reindexa cada estación a una grilla horaria completa (`freq="h"`)
       para que los huecos de la serie queden explícitos como NaN en vez
       de simplemente faltar la fila.
    4. Marca `{col}_imputado` ANTES de imputar, para trazabilidad -- si se
       marcara después ya no se podría distinguir dato real de imputado.
    5. Imputa en 3 pasos, de más a menos confiable:
       a. Interpolación lineal intra-estación, acotada a
          `limite_interpolacion` horas (huecos cortos: la serie es casi
          lineal en esa escala).
       b. Climatología fina: mediana por (estación, mes, hora) -- captura
          estacionalidad anual y ciclo diario.
       c. Climatología gruesa: mediana por (estación, hora) -- fallback
          para combinaciones (mes, hora) sin datos suficientes en el paso
          anterior.

    Parameters
    ----------
    df:
        DataFrame ya normalizado por `cargar_datos`.
    fecha_corte:
        Ver `FECHA_CORTE`. Parametrizado para permitir pruebas, pero el
        valor de producción es el de la constante del módulo.
    limite_interpolacion:
        Ver `LIMITE_INTERPOLACION_HORAS`.
    contaminantes:
        Ver `CONTAMINANTES`.

    Returns
    -------
    DataFrame con grilla horaria completa por estación, columnas
    `{col}_imputado` y sin NaN en `contaminantes`.

    Raises
    ------
    ValueError
        Si tras los 3 pasos de imputación queda algún NaN en
        `contaminantes` (indicaría una estación/hora sin ningún dato
        histórico para calcular climatología).
    """
    df_ventana = df[df["fecha_hora"] >= fecha_corte].copy()
    df_dedup = df_ventana.groupby(
        ["estacion", "codigo_estacion", "fecha_hora"], as_index=False
    )[contaminantes].mean()

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

    for col in contaminantes:
        df_full[col] = df_full.groupby("estacion")[col].transform(
            lambda s: s.interpolate(
                method="linear", limit=limite_interpolacion, limit_direction="both"
            )
        )

    for col in contaminantes:
        perfil_fino = df_full.groupby(["estacion", "mes", "hora"])[col].transform(
            "median"
        )
        df_full[col] = df_full[col].fillna(perfil_fino)

    for col in contaminantes:
        perfil_grueso = df_full.groupby(["estacion", "hora"])[col].transform("median")
        df_full[col] = df_full[col].fillna(perfil_grueso)

    nulos_restantes = df_full[contaminantes].isna().sum()
    if nulos_restantes.any():
        raise ValueError(
            "Quedaron NaN sin imputar tras los 3 pasos de imputación:\n"
            f"{nulos_restantes[nulos_restantes > 0]}"
        )
    assert not df_full[contaminantes].isna().any().any()

    return df_full


def cargar_y_limpiar(ruta_csv: str, **kwargs) -> pd.DataFrame:
    """Combina `cargar_datos` + `limpiar_e_imputar` en una sola llamada.

    Punto de entrada pensado para consumidores externos (models.py,
    forecast.py, app.py) que solo necesitan el DataFrame final limpio y no
    deben preocuparse por el orden de los pasos internos.

    Parameters
    ----------
    ruta_csv:
        Ruta al CSV crudo.
    **kwargs:
        Se reenvían tal cual a `limpiar_e_imputar` (por ejemplo
        `fecha_corte`, `limite_interpolacion`, `contaminantes`) para
        permitir overrides puntuales en pruebas sin tocar las constantes
        del módulo.
    """
    df = cargar_datos(ruta_csv)
    return limpiar_e_imputar(df, **kwargs)


def guardar_parquet(df: pd.DataFrame, ruta_salida: str) -> None:
    """Persiste el DataFrame limpio en formato parquet.

    Parquet en vez de CSV: preserva dtypes (en particular `fecha_hora`
    como datetime y las columnas `{col}_imputado` como bool) para que los
    consumidores no tengan que re-parsear tipos al leerlo.
    """
    df.to_parquet(ruta_salida, index=False)


def reporte_trazabilidad(df: pd.DataFrame, columna: str = "pm_25") -> pd.Series:
    """Calcula el % de filas imputadas por estación para una columna dada.

    Se apoya en la columna `{columna}_imputado` generada por
    `limpiar_e_imputar` (paso 4 del pipeline) para poder auditar, por
    estación, cuánto del dataset final es dato real vs. imputado -- crítico
    para defender la calidad de cualquier modelo entrenado sobre esto.

    Parameters
    ----------
    df:
        DataFrame ya procesado por `limpiar_e_imputar` (debe tener la
        columna `{columna}_imputado`).
    columna:
        Nombre del contaminante a auditar (sin el sufijo `_imputado`).

    Returns
    -------
    Series indexada por `estacion` con el % de filas imputadas.
    """
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
