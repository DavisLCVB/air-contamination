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
from theme import AZUL, ROJO, COLORMAP_CLUSTERS, COLORMAP_DIVERGENTE, aplicar_estilo_mpl

try:  # la lista de contaminantes la define Rol A; hay fallback por si cambia el nombre
    from preprocessing import CONTAMINANTES
except Exception:  # pragma: no cover
    CONTAMINANTES = ["pm_10", "pm_25", "so2", "no2", "o3", "co"]

COL_ESTACION = "estacion"
COL_FECHA = "fecha_hora"
COL_IMPUTADO = "pm_25_imputado"
ECA_PM25_24H = 50  # µg/m³ — D.S. N° 003-2017-MINAM, lectura puntual de 24h (mismo umbral que Panel 2)
ECA_PM25_ANUAL = 25  # µg/m³ — D.S. N° 003-2017-MINAM, promedio de largo plazo (aplica a un promedio histórico multi-año)

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
    más una proyección PCA 2D para poder graficarlo. Devuelve (labels, coords2d, inercia, silueta).
    """
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


def render(df=None):
    """Dibuja el Panel 1: EDA descriptivo + clustering K-means de estaciones."""
    if df is None:
        df = _cargar_df()

    st.subheader("Panel 1 · Análisis exploratorio y clustering de estaciones")
    st.caption("Perfil de contaminación por zona y agrupamiento K-means. "
               "Motiva la tesis de las 'Dos Limas': hay zonas sistemáticamente más "
               "contaminadas que otras.")

    cols = _contaminantes_presentes(df)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Resumen general -------------------------------------------------------------
    with st.container(border=True):
        st.markdown("**📋 Resumen del dataset**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{len(df):,}")
        c2.metric("Estaciones", f"{df[COL_ESTACION].nunique()}")
        fechas = pd.to_datetime(df[COL_FECHA])
        c3.metric("Rango", f"{fechas.min().year}–{fechas.max().year}")
        if COL_IMPUTADO in df.columns:
            c4.metric("PM2.5 imputado", f"{df[COL_IMPUTADO].astype(float).mean() * 100:.1f} %")
        st.caption(
            "Filas y estaciones tras la limpieza; 'PM2.5 imputado' es el % de horas donde "
            "el dato no llegó de la estación y se rellenó (interpolación o climatología), "
            "no un dato medido directamente."
        )

    # --- PM2.5 medio por estación (las 'Dos Limas') ----------------------------------
    with st.container(border=True):
        st.markdown(
            "**🏙️ PM2.5 promedio por estación** (línea sólida = ECA anual 25 µg/m³, la "
            "comparación correcta para un promedio histórico multi-año; línea punteada "
            "tenue = ECA 24h, referencial)"
        )
        pm_est = df.groupby(COL_ESTACION)["pm_25"].mean().sort_values(ascending=False)

        fig1, ax1 = plt.subplots(figsize=(10, 3.8))
        colores = [ROJO if v > ECA_PM25_ANUAL else AZUL for v in pm_est.values]
        ax1.bar(range(len(pm_est)), pm_est.values, color=colores, zorder=2)
        ax1.axhline(ECA_PM25_ANUAL, ls="-", color="gray", linewidth=1.4,
                    label=f"ECA anual = {ECA_PM25_ANUAL} µg/m³")
        ax1.axhline(ECA_PM25_24H, ls=":", color="lightgray", linewidth=1.0,
                    label=f"ECA 24h = {ECA_PM25_24H} µg/m³ (referencial, no aplica directo a un "
                          "promedio histórico)")
        ax1.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
        ax1.set_xticks(range(len(pm_est)))
        ax1.set_xticklabels(pm_est.index, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("PM2.5 (µg/m³)")
        aplicar_estilo_mpl(ax1)
        fig1.tight_layout()
        st.pyplot(fig1, width="stretch")
        n_excede = int((pm_est > ECA_PM25_ANUAL).sum())
        if n_excede > 0:
            st.caption(
                f"Barras en rojo ({n_excede} de {len(pm_est)}): estaciones cuyo promedio histórico "
                f"ya supera el ECA **anual** de {ECA_PM25_ANUAL} µg/m³ (D.S. N° 003-2017-MINAM) — el "
                "estándar correcto para comparar un promedio de varios años. El ECA de 24h "
                f"({ECA_PM25_24H} µg/m³, línea punteada) es el que usa el Panel 2 para clasificar "
                "horas puntuales, no promedios de largo plazo."
            )
        else:
            st.caption(
                f"Ninguna estación supera el ECA **anual** de {ECA_PM25_ANUAL} µg/m³ (línea sólida) "
                f"en promedio histórico — la más cercana es {pm_est.index[0]}, con "
                f"{pm_est.iloc[0]:.1f} µg/m³. El ECA de 24h ({ECA_PM25_24H} µg/m³, línea punteada) "
                "es el que usa el Panel 2 para horas puntuales, no aplica directo aquí. La brecha "
                "entre estaciones sigue siendo la evidencia de que no toda Lima respira el mismo aire."
            )

    # --- Correlación entre contaminantes ---------------------------------------------
    if len(cols) >= 2:
        with st.container(border=True):
            st.markdown("**🔗 Correlación entre contaminantes**")
            st.caption(
                "Cada celda mide qué tan juntos se mueven dos contaminantes (−1 a 1). "
                "Valores cercanos a 1 (rojo) indican que suben y bajan a la vez, típico de "
                "contaminantes con fuentes similares (tráfico, quema); cercanos a 0 (azul) "
                "indican que no están relacionados."
            )
            corr = df[cols].corr()
            fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
            im = ax2.imshow(corr.values, cmap=COLORMAP_DIVERGENTE, vmin=-1, vmax=1)
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
            st.pyplot(fig2, width="stretch")

    # --- Clustering K-means ----------------------------------------------------------
    with st.container(border=True):
        st.markdown("### 🧭 Clustering de estaciones (K-means)")
        col_k, col_inercia, col_silueta = st.columns([2, 1, 1])
        with col_k:
            k = st.slider("Número de clusters (k)", 2, 6, 2,
                          help="k = 2 separa naturalmente las 'Dos Limas' (alta vs baja contaminación).")

        perfil = _perfil_por_estacion(df)
        labels, coords, inercia, silueta = _clusters(perfil, k)

        col_inercia.metric("Inercia", f"{inercia:.2f}",
                            help="Útil para el método del codo al elegir k.")
        col_silueta.metric("Silueta", f"{silueta:.3f}",
                            help="Qué tan bien separado está cada punto de los clusters vecinos "
                                 "(rango −1 a 1; más alto es mejor).")
        st.caption(
            f"⚠️ Inercia y silueta calculadas sobre {len(perfil)} estaciones (una fila por "
            "estación). Con tan pocos puntos, subir k suele dejar clusters de 1-2 miembros y "
            "vuelve la silueta ruidosa — **no es comparable** con la del notebook de EDA, que se "
            "calcula sobre ~515,000 filas a nivel hora-estación, una unidad de análisis distinta."
        )

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sc = ax3.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=COLORMAP_CLUSTERS, s=120,
                          edgecolor="k", zorder=2)
        for i, nombre in enumerate(perfil.index):
            ax3.annotate(nombre, (coords[i, 0], coords[i, 1]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
        ax3.set_xlabel("PCA 1")
        ax3.set_ylabel("PCA 2")
        ax3.set_title(f"Estaciones agrupadas por perfil de contaminación (k={k})")
        aplicar_estilo_mpl(ax3)
        fig3.tight_layout()
        st.pyplot(fig3, width="stretch")
        st.caption(
            "Cada punto es una estación; el color indica su cluster. Los ejes (PCA 1, PCA 2) "
            "son una proyección 2D solo para poder dibujar — el agrupamiento real se calcula "
            "sobre todos los contaminantes a la vez, no sobre estos dos ejes."
        )

        tabla = perfil.copy()
        tabla.insert(0, "cluster", labels)
        st.dataframe(tabla.sort_values("cluster").style.format("{:.1f}", subset=perfil.columns),
                     width="stretch")
        st.caption("Promedio histórico de cada contaminante por estación, agrupado por cluster asignado.")

        with st.expander("Detalles técnicos del clustering"):
            st.markdown(
                f"""
- **Unidad de clustering:** cada estación, representada por el **promedio** de sus
  contaminantes ({', '.join(cols)}).
- **Estandarización:** `StandardScaler` antes de K-means, para que ningún contaminante
  domine solo por tener una escala numérica más grande.
- **Semilla:** `SEED = {SEED}` en K-means y PCA → resultados reproducibles.
- **Visualización:** proyección **PCA 2D** solo para poder dibujar; el agrupamiento se
  hace sobre todas las variables estandarizadas.
"""
            )


# --- Demo aislada: uv run streamlit run src/panel_eda.py ----------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Panel 1 · EDA & Clustering", layout="wide")
    render()
