"""
app.py — Dashboard "Dos Limas, un mismo cielo": calidad del aire en Lima Metropolitana.

Integra los 4 paneles del proyecto en una sola app de Streamlit:

    Panel 1 · EDA + Clustering       -> src/panel_eda.py::render(df)
    Panel 2 · Predicción (RF/XGB)    -> src/panel_predictivo.py::render(df)
    Panel 3 · Serie temporal         -> src/panel_forecast.py::render(df)
    Panel 4 · CRUD + Reporte         -> src/panel_crud.py::render(df)

Contrato único para los cuatro: cada panel expone `render(df=None)` y dibuja su tab.
El df limpio se carga UNA sola vez aquí (cacheado) y se pasa a cada panel.

Paneles que aún no existan muestran un aviso de "en construcción" en lugar de romper
la app, así el deploy sigue funcionando mientras se completan.

Deploy (Streamlit Cloud):  main file = app.py  ·  dependencias = requirements.txt
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import streamlit as st

# Permite `import preprocessing`, `import models`, etc. estén en src/
RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from preprocessing import cargar_y_limpiar  # noqa: E402

RUTA_DATOS = RAIZ / "data" / "air_contamination.csv"

st.set_page_config(page_title="Dos Limas, un mismo cielo", page_icon="🌫️", layout="wide")


@st.cache_data(show_spinner="Cargando y limpiando datos…")
def _df():
    return cargar_y_limpiar(str(RUTA_DATOS))


def _panel(nombre_modulo: str, df, titulo: str):
    """Importa `nombre_modulo.render` y lo ejecuta; si el módulo aún no existe, avisa que está en construcción."""
    try:
        modulo = importlib.import_module(nombre_modulo)
        importlib.reload(modulo)  # recoge cambios sin reiniciar el server durante el desarrollo
        modulo.render(df)
    except ModuleNotFoundError:
        st.subheader(titulo)
        st.info("🚧 Panel en construcción.")
    except Exception as e:  # el panel existe pero falló: no tumbar toda la app
        st.subheader(titulo)
        st.error(f"El panel `{nombre_modulo}` lanzó un error: {e}")
        st.exception(e)


def main():
    col_icono, col_titulo = st.columns([1, 9])
    with col_icono:
        st.markdown("<div style='font-size: 3.2rem; line-height: 1;'>🌫️</div>", unsafe_allow_html=True)
    with col_titulo:
        st.title("Dos Limas, un mismo cielo")
        st.markdown(
            "Clustering y **predicción de calidad del aire (PM2.5)** en Lima Metropolitana · "
            "datos horarios de SENAMHI (10 estaciones, 2014–2020) · pipeline CRISP-DM."
        )

    with st.sidebar:
        st.header("🗺️ Navegación")
        with st.container(border=True):
            st.markdown(
                "- 🔎 **Panel 1** · EDA + Clustering\n"
                "- 🤖 **Panel 2** · Predicción RF/XGBoost\n"
                "- 📈 **Panel 3** · Serie temporal\n"
                "- 🗂️ **Panel 4** · CRUD + Reporte\n"
            )
        st.caption(f"Semilla oficial: `SEED = 96` · datos: `{RUTA_DATOS.name}`")
        st.divider()
        st.caption(
            "💡 Cada panel es independiente: los controles (sliders, selectores) "
            "solo afectan a la pestaña donde están."
        )

    df = _df()
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Filas", f"{len(df):,}")
        m2.metric("Columnas", f"{df.shape[1]}")
        m3.metric("Estaciones", f"{df['estacion'].nunique()}")
    st.divider()

    t1, t2, t3, t4 = st.tabs(
        ["🔎 1 · EDA & Clustering", "🤖 2 · Predicción", "📈 3 · Serie temporal", "🗂️ 4 · CRUD & Reporte"]
    )
    with t1:
        _panel("panel_eda", df, "Panel 1 · EDA & Clustering")
    with t2:
        _panel("panel_predictivo", df, "Panel 2 · Predicción de alta contaminación")
    with t3:
        _panel("panel_forecast", df, "Panel 3 · Serie temporal y pronóstico")
    with t4:
        _panel("panel_crud", df, "Panel 4 · CRUD de consultas & Reporte")


if __name__ == "__main__":
    main()