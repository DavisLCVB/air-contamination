"""
panel_forecast.py — Panel 3 (Series temporales) del dashboard.

Contrato de integración (igual que Rol B / panel_predictivo.py):

    # en app.py
    from panel_forecast import render
    with tab_series:
        render(df)          # df = preprocessing.cargar_y_limpiar(RUTA_DATOS)

- `render(df=None)` carga el df por su cuenta si no se le pasa (para probar aislado).
- Demo aislada:  uv run streamlit run src/panel_forecast.py
- Toda la lógica de series vive en forecast.py; este archivo es solo la UI.

Autor: Rol C.
"""
from __future__ import annotations

import streamlit as st

import forecast as F
from preprocessing import cargar_y_limpiar

_ESTACIONES = [
    "TODAS", "ATE", "CAMPO DE MARTE", "SAN BORJA", "SANTA ANITA",
    "VILLA MARIA DEL TRIUNFO", "HUACHIPA", "SAN JUAN DE LURIGANCHO",
    "SAN MARTIN DE PORRES", "CARABAYLLO", "PUENTE PIEDRA",
]
_FREQS = {"Mensual": "MS", "Semanal": "W", "Diaria": "D"}


@st.cache_data(show_spinner=False)
def _cargar_df():
    return cargar_y_limpiar(str(F.RUTA_DATOS))


@st.cache_data(show_spinner=True, ttl=3600)
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
    st.subheader("Panel 3 · Serie temporal y pronóstico de PM2.5")
    st.caption("Pronóstico ≥ 4 períodos con MAPE y RMSE. Modelos: naive estacional, "
               "Holt-Winters y SARIMA; se elige el de menor MAPE en el hold-out cronológico.")

    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        estacion = st.selectbox("Estación", _ESTACIONES, index=0,
                                help="TODAS = promedio de Lima. Compara ATE/PUENTE PIEDRA "
                                     "(alta) vs CAMPO DE MARTE (baja): las 'Dos Limas'.")
    with c2:
        freq_lbl = st.radio("Frecuencia", list(_FREQS), index=0, horizontal=False,
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

    recortar = st.checkbox(
        "Recortar cola con PM2.5 mayormente imputado (recomendado)", value=True,
        help="Evita pronosticar sobre tramos donde el PM2.5 es casi todo climatología "
             "(relleno estimado, no medido). No afecta a estaciones con dato real hasta el final.",
    )

    freq = _FREQS[freq_lbl]
    try:
        paq = _computar(estacion, freq, periodos_test, horizonte, recortar)
    except ValueError as e:
        st.error(f"No se pudo construir la serie: {e}")
        return

    mejor, cfg = paq["mejor"], paq["config"]
    fila_mejor = paq["tabla"].loc[mejor]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mejor modelo", mejor.replace("_", " ").title())
    m2.metric("MAPE (test)", f"{fila_mejor['mape']:.2f} %")
    m3.metric("RMSE (test)", f"{fila_mejor['rmse']:.2f}")
    m4.metric("Puntos de la serie", f"{len(paq['serie'])}")
    st.caption(
        "MAPE = error porcentual promedio en el hold-out (más bajo es mejor); RMSE = "
        "error en las mismas unidades que PM2.5. 'Mejor modelo' es el de menor MAPE "
        "entre naive estacional, Holt-Winters y SARIMA."
    )

    st.pyplot(F.graficar(paq), width="stretch")
    st.caption(
        "Serie histórica de PM2.5 y, al final, el tramo de test (real vs. predicho por "
        "el modelo ganador) seguido del pronóstico hacia adelante."
    )

    izq, der = st.columns([1, 1])
    with izq:
        st.markdown("**Comparación de modelos** (ordenada por MAPE, menor es mejor)")
        st.caption("El valor resaltado en verde es el mínimo de cada métrica entre los tres modelos.")
        tabla = paq["tabla"].rename(columns={"mape": "MAPE %", "rmse": "RMSE", "mae": "MAE"})
        st.dataframe(tabla.style.format("{:.3f}").highlight_min(axis=0, color="#a3be8c55"),
                     width="stretch")
    with der:
        st.markdown(f"**Pronóstico — próximos {cfg['horizonte']} períodos**")
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


# --- Demo aislada: uv run streamlit run src/panel_forecast.py -----------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Panel 3 · Pronóstico PM2.5", layout="wide")
    render()
