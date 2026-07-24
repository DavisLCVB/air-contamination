from __future__ import annotations

import pandas as pd

from core.preprocessing import CONTAMINANTES, SEED

ECA_PM25_24H = 50  # µg/m³ -- D.S. N° 003-2017-MINAM, lectura puntual de 24h
ECA_PM25_ANUAL = 25  # µg/m³ -- D.S. N° 003-2017-MINAM, promedio de largo plazo


def contaminantes_presentes(df: pd.DataFrame) -> list[str]:
    return [c for c in CONTAMINANTES if c in df.columns]


def perfil_por_estacion(df: pd.DataFrame) -> pd.DataFrame:
    cols = contaminantes_presentes(df)
    return df.groupby("estacion")[cols].mean()


def clusters(perfil: pd.DataFrame, k: int):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    X = StandardScaler().fit_transform(perfil.values)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    coords = PCA(n_components=2, random_state=SEED).fit_transform(X)
    silueta = float(silhouette_score(X, labels))
    return labels, coords, float(km.inertia_), silueta
