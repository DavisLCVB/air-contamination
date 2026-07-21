"""
app.py — Dashboard "Dos Limas, un mismo cielo": calidad del aire en Lima Metropolitana.

Integra los 4 paneles del proyecto en una sola app de Streamlit:

    Panel 1 · EDA + Clustering       -> src/panel_eda.py::render(df)
    Panel 2 · Predicción (RF/XGB)    -> src/panel_predictivo.py::render(df)
    Panel 3 · Serie temporal         -> src/panel_forecast.py::render(df)
    Panel 4 · CRUD + Reporte         -> src/panel_crud.py::render(df)

Contrato único para los cuatro: cada panel expone `render(df=None)` y dibuja su
contenido. La navegación entre paneles vive en botones del sidebar (no tabs);
el panel activo se guarda en `st.session_state["panel_activo"]`. El df limpio
se carga UNA sola vez aquí (cacheado) y se pasa al panel activo.

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
import theme  # noqa: E402

RUTA_DATOS = RAIZ / "data" / "air_contamination.csv"

st.set_page_config(page_title="Dos Limas, un mismo cielo", page_icon=":material/analytics:", layout="wide")

# (módulo, ícono+etiqueta del botón de navegación, título de fallback si el panel falla)
PANELES = [
    ("panel_eda", ":material/query_stats: 1 · EDA & Clustering", "Panel 1 · EDA & Clustering"),
    ("panel_predictivo", ":material/smart_toy: 2 · Predicción", "Panel 2 · Predicción de alta contaminación"),
    ("panel_forecast", ":material/show_chart: 3 · Serie temporal", "Panel 3 · Serie temporal y pronóstico"),
    ("panel_crud", ":material/folder_open: 4 · CRUD & Reporte", "Panel 4 · CRUD de consultas & Reporte"),
]


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
        st.info("Panel en construcción.", icon=":material/construction:")
    except Exception as e:  # el panel existe pero falló: no tumbar toda la app
        st.subheader(titulo)
        st.error(f"El panel `{nombre_modulo}` lanzó un error: {e}")
        st.exception(e)


def main():
    theme.inyectar_css()

    if "panel_activo" not in st.session_state:
        st.session_state.panel_activo = PANELES[0][0]

    st.title(":material/analytics: Dos Limas, un mismo cielo")
    st.markdown(
        "Clustering y **predicción de calidad del aire (PM2.5)** en Lima Metropolitana · "
        "datos horarios de SENAMHI (10 estaciones, 2014–2020) · pipeline CRISP-DM."
    )

    with st.sidebar:
        st.header(":material/map: Navegación")
        with st.container(border=True):
            for modulo, etiqueta, _ in PANELES:
                activo = st.session_state.panel_activo == modulo
                if st.button(etiqueta, key=f"nav_{modulo}", width="stretch",
                             type="primary" if activo else "secondary"):
                    st.session_state.panel_activo = modulo
                    st.rerun()
        st.caption(f"Semilla oficial: `SEED = 96` · datos: `{RUTA_DATOS.name}`")

        st.divider()
        st.caption(
            ":material/lightbulb: Cada panel es independiente: los controles (sliders, "
            "selectores) solo afectan al panel donde están."
        )

    df = _df()
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Filas", f"{len(df):,}")
        m2.metric("Columnas", f"{df.shape[1]}")
        m3.metric("Estaciones", f"{df['estacion'].nunique()}")
    st.divider()

    modulo_activo, _, titulo_activo = next(
        (p for p in PANELES if p[0] == st.session_state.panel_activo), PANELES[0]
    )
    _panel(modulo_activo, df, titulo_activo)


if __name__ == "__main__":
    main()