"""
panel_eda.py — Panel 1 (EDA + Clustering) del dashboard.

Contrato de integración (igual que Rol B / panel_predictivo.py y Rol C / panel_forecast.py):

    # en app.py
    from panel_eda import render
    with tab_eda:
        render(df)          # df = preprocessing.cargar_y_limpiar(RUTA_DATOS)

- `render(df=None)` carga el df por su cuenta si no se le pasa (para probar aislado).
- Demo aislada:  uv run streamlit run src/panel_eda.py
- La limpieza y la semilla oficial (SEED) vienen de Rol A (preprocessing.py).

Autor: Rol A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import cargar_y_limpiar, SEED

try:  # la lista de contaminantes la define Rol A; hay fallback por si cambia el nombre
    from preprocessing import CONTAMINANTES
except Exception:  # pragma: no cover
    CONTAMINANTES = ["pm_10", "pm_25", "so2", "no2", "o3", "co"]

COL_ESTACION = "estacion"
COL_FECHA = "fecha_hora"
COL_IMPUTADO = "pm_25_imputado"
ECA_PM25 = 50  # µg/m³ (mismo umbral que la etiqueta de Rol B)

# Ruta de datos: se reutiliza la de forecast.py si está disponible, para no duplicar
try:
    from forecast import RUTA_DATOS
except Exception:  # pragma: no cover
    from pathlib import Path
    RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "air_contamination.csv"


@st.cache_data(show_spinner=False)
def _cargar_df():
    return cargar_y_limpiar(str(RUTA_DATOS))


def _contaminantes_presentes(df: pd.DataFrame) -> list[str]:
    """Contaminantes de la lista oficial que realmente están en el df."""
    return [c for c in CONTAMINANTES if c in df.columns]


@st.cache_data(show_spinner=False)
def _perfil_por_estacion(df: pd.DataFrame) -> pd.DataFrame:
    """Promedio de cada contaminante por estación — la 'huella' de cada zona."""
    cols = _contaminantes_presentes(df)
    return df.groupby(COL_ESTACION)[cols].mean()


@st.cache_data(show_spinner=True)
def _clusters(perfil: pd.DataFrame, k: int):
    """
    Clustering K-means de las estaciones sobre su perfil de contaminación (estandarizado),
    más una proyección PCA 2D para poder graficarlo. Devuelve (labels, coords2d, inercia).
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(perfil.values)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    coords = PCA(n_components=2, random_state=SEED).fit_transform(X)
    return labels, coords, float(km.inertia_)


def render(df=None):
    """Dibuja el Panel 1: EDA descriptivo + clustering K-means de estaciones."""
    if df is None:
        df = _cargar_df()

    st.subheader("Panel 1 · Análisis exploratorio y clustering de estaciones")
    st.caption("Rol A — perfil de contaminación por zona y agrupamiento K-means. "
               "Motiva la tesis de las 'Dos Limas': hay zonas sistemáticamente más "
               "contaminadas que otras.")

    cols = _contaminantes_presentes(df)

    # --- Resumen general -------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Estaciones", f"{df[COL_ESTACION].nunique()}")
    fechas = pd.to_datetime(df[COL_FECHA])
    c3.metric("Rango", f"{fechas.min().year}–{fechas.max().year}")
    if COL_IMPUTADO in df.columns:
        c4.metric("PM2.5 imputado", f"{df[COL_IMPUTADO].astype(float).mean() * 100:.1f} %")

    # --- PM2.5 medio por estación (las 'Dos Limas') ----------------------------------
    st.markdown("**PM2.5 promedio por estación** (línea roja = ECA 50 µg/m³)")
    pm_est = df.groupby(COL_ESTACION)["pm_25"].mean().sort_values(ascending=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(10, 3.8))
    colores = ["#bf616a" if v > ECA_PM25 else "#5e81ac" for v in pm_est.values]
    ax1.bar(range(len(pm_est)), pm_est.values, color=colores)
    ax1.axhline(ECA_PM25, ls=":", color="gray")
    ax1.set_xticks(range(len(pm_est)))
    ax1.set_xticklabels(pm_est.index, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("PM2.5 (µg/m³)")
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)

    # --- Correlación entre contaminantes ---------------------------------------------
    if len(cols) >= 2:
        st.markdown("**Correlación entre contaminantes**")
        corr = df[cols].corr()
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
        im = ax2.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(cols)))
        ax2.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(len(cols)))
        ax2.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax2.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color="black")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # --- Clustering K-means ----------------------------------------------------------
    st.markdown("### Clustering de estaciones (K-means)")
    k = st.slider("Número de clusters (k)", 2, 6, 2,
                  help="k = 2 separa naturalmente las 'Dos Limas' (alta vs baja contaminación).")

    perfil = _perfil_por_estacion(df)
    labels, coords, inercia = _clusters(perfil, k)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sc = ax3.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Set1", s=120, edgecolor="k")
    for i, nombre in enumerate(perfil.index):
        ax3.annotate(nombre, (coords[i, 0], coords[i, 1]), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    ax3.set_xlabel("PCA 1")
    ax3.set_ylabel("PCA 2")
    ax3.set_title(f"Estaciones agrupadas por perfil de contaminación (k={k})")
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    tabla = perfil.copy()
    tabla.insert(0, "cluster", labels)
    st.dataframe(tabla.sort_values("cluster").style.format("{:.1f}", subset=perfil.columns),
                 use_container_width=True)

    with st.expander("Notas de modelado (Rol A)"):
        st.markdown(
            f"""
- **Unidad de clustering:** cada estación, representada por el **promedio** de sus
  contaminantes ({', '.join(cols)}).
- **Estandarización:** `StandardScaler` antes de K-means (a diferencia de RF/XGBoost de
  Rol B, que son invariantes a escala).
- **Semilla:** `SEED = {SEED}` en K-means y PCA → resultados reproducibles.
- **Visualización:** proyección **PCA 2D** solo para poder dibujar; el agrupamiento se
  hace sobre todas las variables estandarizadas.
- **Inercia (k={k}):** {inercia:.2f} — útil para el método del codo al elegir k.
"""
        )


# --- Demo aislada: uv run streamlit run src/panel_eda.py ----------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Panel 1 · EDA & Clustering", layout="wide")
    render()
