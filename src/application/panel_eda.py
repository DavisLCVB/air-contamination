"""
panel_eda.py — Panel 1 (EDA + Clustering) del dashboard.

Expone `render(df=None)`: dibuja el perfil de contaminación por estación y el
clustering K-means. Si no se le pasa `df`, lo carga por su cuenta (demo aislada:
uv run streamlit run src/application/panel_eda.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core.clustering import ECA_PM25_24H, ECA_PM25_ANUAL, clusters, perfil_por_estacion
from core.forecast import RUTA_DATOS
from core.preprocessing import SEED, cargar_y_limpiar

from application import theme

COL_ESTACION = "estacion"
COL_FECHA = "fecha_hora"
COL_IMPUTADO = "pm_25_imputado"


@st.cache_data(show_spinner=False)
def _cargar_df():
    return cargar_y_limpiar(str(RUTA_DATOS))


@st.cache_data(show_spinner=False)
def _perfil_por_estacion(df: pd.DataFrame) -> pd.DataFrame:
    return perfil_por_estacion(df)


@st.cache_data(show_spinner=True)
def _clusters(perfil: pd.DataFrame, k: int):
    return clusters(perfil, k)


def render(df=None):
    """Dibuja el Panel 1: EDA descriptivo + clustering K-means de estaciones."""
    if df is None:
        df = _cargar_df()

    st.subheader("Panel 1 · Análisis exploratorio y clustering de estaciones")
    st.caption("Perfil de contaminación por zona y agrupamiento K-means. "
               "Motiva la tesis de las 'Dos Limas': hay zonas sistemáticamente más "
               "contaminadas que otras.")

    cols = [c for c in ["pm_10", "pm_25", "so2", "no2", "o3", "co"] if c in df.columns]
    p = theme.paleta()

    # --- Resumen general -------------------------------------------------------------
    with st.container(border=True):
        st.subheader(":material/summarize: Resumen del dataset")
        fechas = pd.to_datetime(df[COL_FECHA])
        with st.container(horizontal=True):
            st.metric("Filas", f"{len(df):,}", border=True)
            st.metric("Estaciones", f"{df[COL_ESTACION].nunique()}", border=True)
            st.metric("Rango", f"{fechas.min().year}–{fechas.max().year}", border=True)
            if COL_IMPUTADO in df.columns:
                st.metric("PM2.5 imputado", f"{df[COL_IMPUTADO].astype(float).mean() * 100:.1f} %",
                          border=True)
        st.caption(
            "Filas y estaciones tras la limpieza; 'PM2.5 imputado' es el % de horas donde "
            "el dato no llegó de la estación y se rellenó (interpolación o climatología), "
            "no un dato medido directamente."
        )

    # --- PM2.5 medio por estación (las 'Dos Limas') ----------------------------------
    with st.container(border=True):
        st.subheader(":material/location_city: PM2.5 promedio por estación")

        pm_est = df.groupby(COL_ESTACION)["pm_25"].mean().sort_values(ascending=False)

        etiqueta_excede = f"> ECA anual ({ECA_PM25_ANUAL} µg/m³)"
        etiqueta_normal = f"≤ ECA anual ({ECA_PM25_ANUAL} µg/m³)"
        datos_barra = pm_est.reset_index()
        datos_barra.columns = [COL_ESTACION, "pm25"]
        datos_barra["categoria"] = np.where(datos_barra["pm25"] > ECA_PM25_ANUAL,
                                             etiqueta_excede, etiqueta_normal)

        barras = alt.Chart(datos_barra).mark_bar().encode(
            x=alt.X(f"{COL_ESTACION}:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"),
            color=alt.Color("categoria:N", title=None,
                             scale=alt.Scale(domain=[etiqueta_excede, etiqueta_normal],
                                             range=[p["ROJO"], p["AZUL"]])),
            tooltip=[alt.Tooltip(f"{COL_ESTACION}:N", title="Estación"),
                     alt.Tooltip("pm25:Q", title="PM2.5 promedio", format=".1f")],
        )
        linea_anual = alt.Chart(pd.DataFrame({"y": [ECA_PM25_ANUAL]})).mark_rule(
            color=p["REFERENCIA"], strokeWidth=1.4).encode(y="y:Q")
        linea_24h = alt.Chart(pd.DataFrame({"y": [ECA_PM25_24H]})).mark_rule(
            color=p["REFERENCIA_TENUE"], strokeDash=[5, 3]).encode(y="y:Q")

        chart1 = (barras + linea_anual + linea_24h).properties(height=340)
        n_excede = int((pm_est > ECA_PM25_ANUAL).sum())
        if n_excede > 0:
            texto_barras = (
                f"Barras en rojo ({n_excede} de {len(pm_est)}): estaciones cuyo promedio histórico "
                f"ya supera el ECA **anual** de {ECA_PM25_ANUAL} µg/m³ (D.S. N° 003-2017-MINAM) — el "
                "estándar correcto para comparar un promedio de varios años. El ECA de 24h "
                f"({ECA_PM25_24H} µg/m³, línea punteada) es el que usa el Panel 2 para clasificar "
                "horas puntuales, no promedios de largo plazo."
            )
        else:
            texto_barras = (
                f"Ninguna estación supera el ECA **anual** de {ECA_PM25_ANUAL} µg/m³ (línea sólida) "
                f"en promedio histórico — la más cercana es {pm_est.index[0]}, con "
                f"{pm_est.iloc[0]:.1f} µg/m³. El ECA de 24h ({ECA_PM25_24H} µg/m³, línea punteada) "
                "es el que usa el Panel 2 para horas puntuales, no aplica directo aquí. La brecha "
                "entre estaciones sigue siendo la evidencia de que no toda Lima respira el mismo aire."
            )

        col_chart, col_texto = st.columns([2, 1])
        with col_chart:
            st.altair_chart(theme.aplicar_estilo_altair(chart1), theme=None, width="stretch")
        with col_texto:
            st.caption(
                "Compara el promedio histórico de cada estación contra el ECA anual "
                f"(línea sólida, {ECA_PM25_ANUAL} µg/m³) y el ECA de 24h (línea punteada, "
                f"{ECA_PM25_24H} µg/m³, solo referencial para un promedio multi-año)."
            )
            st.caption(texto_barras)

    # --- Correlación entre contaminantes ---------------------------------------------
    if len(cols) >= 2:
        with st.container(border=True):
            st.subheader(":material/link: Correlación entre contaminantes")

            corr = df[cols].corr()
            corr_largo = (
                corr.reset_index()
                .melt(id_vars="index", var_name="contaminante_2", value_name="correlacion")
                .rename(columns={"index": "contaminante_1"})
            )

            heatmap = alt.Chart(corr_largo).mark_rect().encode(
                x=alt.X("contaminante_2:N", title=None, sort=cols),
                y=alt.Y("contaminante_1:N", title=None, sort=cols),
                color=alt.Color("correlacion:Q", title="Correlación",
                                 scale=alt.Scale(domain=[-1, 0, 1],
                                                 range=[p["AZUL"], p["SUPERFICIE_2"], p["ROJO"]])),
                tooltip=[alt.Tooltip("contaminante_1:N", title="Contaminante 1"),
                         alt.Tooltip("contaminante_2:N", title="Contaminante 2"),
                         alt.Tooltip("correlacion:Q", title="Correlación", format=".2f")],
            )
            texto = alt.Chart(corr_largo).mark_text(fontSize=11).encode(
                x=alt.X("contaminante_2:N", sort=cols),
                y=alt.Y("contaminante_1:N", sort=cols),
                text=alt.Text("correlacion:Q", format=".2f"),
                color=alt.condition("abs(datum.correlacion) > 0.6",
                                     alt.value(p["FONDO"]), alt.value(p["TEXTO"])),
            )
            chart2 = (heatmap + texto).properties(height=380)

            col_chart, col_texto = st.columns([2, 1])
            with col_chart:
                st.altair_chart(theme.aplicar_estilo_altair(chart2), theme=None, width="stretch")
            with col_texto:
                st.caption(
                    "Cada celda mide qué tan juntos se mueven dos contaminantes (−1 a 1). "
                    "Valores cercanos a 1 (rojo) indican que suben y bajan a la vez, típico de "
                    "contaminantes con fuentes similares (tráfico, quema); cercanos a 0 (azul) "
                    "indican que no están relacionados."
                )

    # --- Clustering K-means ----------------------------------------------------------
    with st.container(border=True):
        st.subheader(":material/scatter_plot: Clustering de estaciones (K-means)")
        st.caption(
            "Agrupa las estaciones según su perfil de contaminación (no según ubicación "
            "geográfica) para verificar si las 'Dos Limas' emergen como clusters propios."
        )
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
            f":material/warning: Inercia y silueta calculadas sobre {len(perfil)} estaciones (una fila por "
            "estación). Con tan pocos puntos, subir k suele dejar clusters de 1-2 miembros y "
            "vuelve la silueta ruidosa — **no es comparable** con la del notebook de EDA, que se "
            "calcula sobre ~515,000 filas a nivel hora-estación, una unidad de análisis distinta."
        )

        datos_cluster = pd.DataFrame({
            "estacion": perfil.index,
            "pca1": coords[:, 0],
            "pca2": coords[:, 1],
            "cluster": labels.astype(str),
        })
        scatter = alt.Chart(datos_cluster).mark_circle(
            size=180, opacity=0.9, stroke=p["TEXTO"], strokeWidth=0.8
        ).encode(
            x=alt.X("pca1:Q", title="PCA 1"),
            y=alt.Y("pca2:Q", title="PCA 2"),
            color=alt.Color("cluster:N", title="Cluster",
                             scale=alt.Scale(range=theme.lista_colores_clusters())),
            tooltip=[alt.Tooltip("estacion:N", title="Estación"),
                     alt.Tooltip("cluster:N", title="Cluster")],
        )
        etiquetas = alt.Chart(datos_cluster).mark_text(
            dx=8, dy=-8, fontSize=9, align="left", color=p["TEXTO"]
        ).encode(x="pca1:Q", y="pca2:Q", text="estacion:N")

        chart3 = (scatter + etiquetas).properties(
            height=380, title=f"Estaciones agrupadas por perfil de contaminación (k={k})"
        )
        col_chart, col_texto = st.columns([2, 1])
        with col_chart:
            st.altair_chart(theme.aplicar_estilo_altair(chart3), theme=None, width="stretch")
        with col_texto:
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


if __name__ == "__main__":
    st.set_page_config(page_title="Panel 1 · EDA & Clustering", layout="wide")
    render()
