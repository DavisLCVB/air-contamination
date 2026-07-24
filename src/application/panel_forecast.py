"""
panel_forecast.py — Panel 3 (Series temporales) del dashboard.

Expone `render(df=None)`: toda la lógica de series vive en core.forecast, este
archivo es solo la UI (demo aislada: uv run streamlit run src/application/panel_forecast.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

import core.forecast as F
from core.preprocessing import cargar_y_limpiar

from application import theme

_ESTACIONES = [
    "TODAS", "ATE", "CAMPO DE MARTE", "SAN BORJA", "SANTA ANITA",
    "VILLA MARIA DEL TRIUNFO", "HUACHIPA", "SAN JUAN DE LURIGANCHO",
    "SAN MARTIN DE PORRES", "CARABAYLLO", "PUENTE PIEDRA",
]
_FREQS = {"Mensual": "MS", "Semanal": "W", "Diaria": "D"}


@st.cache_data(show_spinner=False)
def _cargar_df():
    return cargar_y_limpiar(str(F.RUTA_DATOS))


@st.cache_data(show_spinner=False, ttl=3600)
def _computar(estacion, freq, periodos_test, horizonte, recortar):
    """Corre serie -> comparar -> mejor -> futuro y devuelve solo objetos ligeros (cacheable)."""
    df = _cargar_df()
    paq = F.serie_y_pronostico(df, estacion, freq, periodos_test, horizonte, recortar)
    # se descarta el objeto de modelo ajustado (no hace falta para dibujar y aligera el caché)
    resultados = {
        k: {"yhat": v["yhat"], "mape": v["mape"], "rmse": v["rmse"], "mae": v["mae"]}
        for k, v in paq["resultados"].items()
    }
    paq["resultados"] = resultados
    return paq


def render(df=None):
    """Dibuja el Panel 3. `df` se acepta por el contrato de app.py pero el cómputo se cachea aparte."""
    st.subheader(":material/show_chart: Panel 3 · Serie temporal y pronóstico de PM2.5")
    st.caption("Pronóstico ≥ 4 períodos con MAPE y RMSE. Modelos: naive estacional, "
               "Holt-Winters y SARIMA; se elige el de menor MAPE en el hold-out cronológico.")

    with st.container(border=True):
        st.subheader(":material/tune: Configuración del pronóstico")
        st.caption("Define qué serie modelar y con qué parámetros evaluar el pronóstico.")
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
        with c1:
            estacion = st.selectbox("Estación", _ESTACIONES, index=0,
                                    help="TODAS = promedio de Lima. Compara ATE/PUENTE PIEDRA "
                                         "(alta) vs CAMPO DE MARTE (baja): las 'Dos Limas'.")
        with c2:
            freq_lbl = st.segmented_control("Frecuencia", list(_FREQS), default=list(_FREQS)[0],
                                            help="A qué intervalo se agrega el PM2.5 antes de modelar "
                                                 "la serie (hora a hora es demasiado ruidoso para pronosticar).")
        with c3:
            horizonte = st.slider("Horizonte (períodos a pronosticar)", 4, 12, F.HORIZONTE,
                                  help="Cuántos períodos hacia el futuro se proyectan, en la "
                                       "unidad elegida en 'Frecuencia'.")
        with c4:
            periodos_test = st.slider("Ventana de test", 4, 18, F.PERIODOS_TEST,
                                      help="Cuántos períodos finales de la serie se separan como "
                                           "hold-out para medir MAPE/RMSE antes de pronosticar el futuro.")

        recortar = st.toggle(
            "Recortar cola con PM2.5 mayormente imputado (recomendado)", value=True,
            help="Evita pronosticar sobre tramos donde el PM2.5 es casi todo climatología "
                 "(relleno estimado, no medido). No afecta a estaciones con dato real hasta el final.",
        )

    freq = _FREQS[freq_lbl]
    try:
        with st.skeleton(height=160):
            paq = _computar(estacion, freq, periodos_test, horizonte, recortar)
    except ValueError as e:
        st.error(f"No se pudo construir la serie: {e}")
        return

    mejor, cfg = paq["mejor"], paq["config"]
    fila_mejor = paq["tabla"].loc[mejor]

    with st.container(border=True):
        with st.container(horizontal=True):
            st.metric("Mejor modelo", mejor.replace("_", " ").title(), border=True)
            st.metric("MAPE (test)", f"{fila_mejor['mape']:.2f} %", border=True)
            st.metric("RMSE (test)", f"{fila_mejor['rmse']:.2f}", border=True)
            st.metric("Puntos de la serie", f"{len(paq['serie'])}", border=True)
        st.caption(
            "MAPE = error porcentual promedio en el hold-out (más bajo es mejor); RMSE = "
            "error en las mismas unidades que PM2.5. 'Mejor modelo' es el de menor MAPE "
            "entre naive estacional, Holt-Winters y SARIMA."
        )

        st.subheader(":material/show_chart: Serie histórica y pronóstico")

        p = theme.paleta()
        serie = paq["serie"]
        yhat_test = paq["resultados"][mejor]["yhat"]
        fut = paq["futuro"]

        etiqueta_ajuste = f"Ajuste test ({mejor})"
        colores_serie = {"Histórico": p["TEXTO"], etiqueta_ajuste: p["ROJO"], "Pronóstico": p["AZUL"]}
        df_lineas = pd.concat([
            pd.DataFrame({"fecha": serie.index, "valor": serie.values, "serie": "Histórico"}),
            pd.DataFrame({"fecha": yhat_test.index, "valor": yhat_test.values, "serie": etiqueta_ajuste}),
            pd.DataFrame({"fecha": fut.index, "valor": fut["yhat"].values, "serie": "Pronóstico"}),
        ], ignore_index=True)

        capas = []
        if "lo" in fut.columns:
            df_banda = pd.DataFrame({"fecha": fut.index, "lo": fut["lo"].values, "hi": fut["hi"].values})
            capas.append(
                alt.Chart(df_banda).mark_area(opacity=0.18, color=p["AZUL"])
                .encode(x="fecha:T", y="lo:Q", y2="hi:Q")
            )

        lineas = alt.Chart(df_lineas).mark_line().encode(
            x=alt.X("fecha:T", title=None),
            y=alt.Y("valor:Q", title="PM2.5 (µg/m³)"),
            color=alt.Color("serie:N", title=None,
                             scale=alt.Scale(domain=list(colores_serie), range=list(colores_serie.values()))),
            strokeDash=alt.StrokeDash("serie:N",
                             scale=alt.Scale(domain=list(colores_serie), range=[[1, 0], [6, 3], [1, 0]])),
            tooltip=[alt.Tooltip("fecha:T", title="Fecha"), alt.Tooltip("serie:N", title="Serie"),
                     alt.Tooltip("valor:Q", title="PM2.5", format=".1f")],
        )
        capas.append(lineas)

        titulo = (f"PM2.5 — {cfg['estacion']} · mejor: {mejor} · "
                  f"MAPE {fila_mejor['mape']:.1f}% · RMSE {fila_mejor['rmse']:.1f}")
        chart_serie = alt.layer(*capas).properties(height=340, title=titulo)

        col_chart, col_texto = st.columns([2, 1])
        with col_chart:
            st.altair_chart(theme.aplicar_estilo_altair(chart_serie), theme=None, width="stretch")
        with col_texto:
            st.caption(
                "Serie histórica de PM2.5 y, al final, el tramo de test (real vs. predicho por "
                "el modelo ganador) seguido del pronóstico hacia adelante."
            )

    izq, der = st.columns([1, 1])
    with izq:
        with st.container(border=True):
            st.subheader(":material/bar_chart: Comparación de modelos")
            st.caption("Ordenada por MAPE (menor es mejor); el valor resaltado en verde es el "
                       "mínimo de cada métrica entre los tres modelos.")
            tabla = paq["tabla"].rename(columns={"mape": "MAPE %", "rmse": "RMSE", "mae": "MAE"})
            st.dataframe(tabla.style.format("{:.3f}").highlight_min(axis=0, color="#a3be8c55"),
                         width="stretch")
    with der:
        with st.container(border=True):
            st.subheader(f":material/insights: Pronóstico — próximos {cfg['horizonte']} períodos")
            st.caption("Valores proyectados por el modelo ganador, reajustado sobre toda la serie disponible.")
            fut = paq["futuro"].copy()
            fut.index = fut.index.strftime("%Y-%m-%d")
            st.dataframe(fut.round(2), width="stretch")

    with st.expander("Detalles técnicos del pronóstico"):
        st.markdown(
            f"""
- **Objetivo:** `pm_25` promediado a frecuencia **{freq_lbl.lower()}** (período estacional
  m = {cfg['estacionalidad']}).
- **Split cronológico** (sin barajar): últimos **{cfg['periodos_test']}** períodos como test;
  el modelo elegido se reajusta sobre la serie completa para el pronóstico futuro.
- **Métricas:** MAPE (%) y RMSE, más MAE como apoyo. Se elige por menor MAPE.
- **Baseline honesto:** el *naive estacional* (copiar el último ciclo) sirve de piso;
  un modelo solo se justifica si le gana.
- **Sin fuga:** el pronóstico usa únicamente el pasado de la propia serie.
- Cifras auditables en `models/forecast_metrics.json`.
"""
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Panel 3 · Pronóstico PM2.5", layout="wide")
    render()
