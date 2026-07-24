from __future__ import annotations

import numpy as np
import pandas as pd

import core.forecast as forecast
from core.clustering import clusters, perfil_por_estacion

N_MIN = 24  # meses mínimos para confiar en la pendiente de tendencia

# Zona muerta: con series de ~5-6 años, una pendiente dentro de ±0.01 µg/m³/mes es
# indistinguible del ruido natural de la serie -> se trata como "sin cambio
# confirmado", no como mejora.
UMBRAL_MEJORA = -0.01  # µg/m³ por mes

# Coordenadas aproximadas (centro del distrito/localidad, fuente pública) --
# referenciales para el mapa, no son las coordenadas exactas del instrumento SENAMHI.
COORDENADAS = {
    "ATE": (-12.0103, -76.8700),
    "CAMPO DE MARTE": (-12.0681, -77.0419),
    "SAN BORJA": (-12.1000, -77.0170),
    "SANTA ANITA": (-12.0432, -76.9631),
    "VILLA MARIA DEL TRIUNFO": (-12.1570, -76.9310),
    "HUACHIPA": (-11.9988, -76.9307),
    "SAN JUAN DE LURIGANCHO": (-12.0330, -77.0170),
    "SAN MARTIN DE PORRES": (-12.0278, -77.0433),
    "CARABAYLLO": (-11.8500, -77.0330),
    "PUENTE PIEDRA": (-11.8750, -77.0653),
}


def pendiente(serie: pd.Series) -> float:
    if len(serie) < N_MIN:
        return float("nan")
    x = np.arange(len(serie))
    return float(np.polyfit(x, serie.values, 1)[0])


def tabla_prioridad(df: pd.DataFrame) -> pd.DataFrame:
    perfil = perfil_por_estacion(df)
    labels, _coords, _inercia, _silueta = clusters(perfil, 2)

    medias_cluster = perfil["pm_25"].groupby(labels).mean()
    label_alta = medias_cluster.idxmax()

    filas = []
    for i, estacion in enumerate(perfil.index):
        serie = forecast.construir_serie(df, estacion=estacion, freq="MS")
        filas.append({
            "estacion": estacion,
            "pm25": float(perfil.loc[estacion, "pm_25"]),
            "cluster_alta": bool(labels[i] == label_alta),
            "pendiente": pendiente(serie),
            "n_meses": len(serie),
        })

    tabla = pd.DataFrame(filas)
    tabla["criterio_severidad"] = tabla["cluster_alta"]
    tabla["criterio_trayectoria"] = tabla["pendiente"] >= UMBRAL_MEJORA
    tabla["prioridad"] = tabla["criterio_severidad"] & tabla["criterio_trayectoria"]
    return tabla.sort_values(["prioridad", "pm25"], ascending=[False, False]).reset_index(drop=True)
