"""
app.py — Dashboard integrado "Dos Limas, un mismo cielo" (Rol C, infraestructura).

Integra los 4 paneles del proyecto en una sola app de Streamlit:

    Panel 1 · EDA + Clustering       (Rol A)  -> src/panel_eda.py::render(df)
    Panel 2 · Predicción (RF/XGB)    (Rol B)  -> src/panel_predictivo.py::render(df)
    Panel 3 · Serie temporal         (Rol C)  -> src/panel_forecast.py::render(df)
    Panel 4 · CRUD + Reporte         (Rol D)  -> src/panel_crud.py::render(df)

Contrato único para los cuatro: cada panel expone `render(df=None)` y dibuja su tab.
El df limpio se carga UNA sola vez aquí (cacheado) y se pasa a cada panel, de modo que
la limpieza de Rol A se ejecuta una única vez por sesión.

Paneles que aún no existan (Rol A/D en curso) muestran un aviso con el contrato a cumplir
en lugar de romper la app — así el deploy funciona apenas cada rol suelte su `render`.

Deploy (Streamlit Cloud):  main file = app.py  ·  dependencias = requirements.txt

Autor: Rol C.
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


@st.cache_data(show_spinner="Cargando y limpiando datos (Rol A)…")
def _df():
    return cargar_y_limpiar(str(RUTA_DATOS))


def _panel(nombre_modulo: str, df, titulo: str, rol: str, entregable: str):
    """Importa `nombre_modulo.render` y lo ejecuta; si no existe aún, muestra el contrato."""
    try:
        modulo = importlib.import_module(nombre_modulo)
        importlib.reload(modulo)  # recoge cambios sin reiniciar el server durante el desarrollo
        modulo.render(df)
    except ModuleNotFoundError:
        st.subheader(titulo)
        st.info(
            f"**{rol}** todavía no ha integrado este panel.\n\n"
            f"Para que aparezca aquí, crear `src/{nombre_modulo}.py` con una función "
            f"`render(df=None)` que dibuje el contenido del panel.\n\n"
            f"Entregable: {entregable}."
        )
    except Exception as e:  # el panel existe pero falló: no tumbar toda la app
        st.subheader(titulo)
        st.error(f"El panel `{nombre_modulo}` lanzó un error: {e}")
        st.exception(e)


def main():
    st.title("🌫️ Dos Limas, un mismo cielo")
    st.markdown(
        "Clustering y **predicción de calidad del aire (PM2.5)** en Lima Metropolitana · "
        "datos horarios de SENAMHI (10 estaciones, 2014–2020) · pipeline CRISP-DM."
    )

    with st.sidebar:
        st.header("Proyecto")
        st.markdown(
            "- **Panel 1** · EDA + Clustering — Rol A\n"
            "- **Panel 2** · Predicción RF/XGBoost — Rol B\n"
            "- **Panel 3** · Serie temporal — Rol C\n"
            "- **Panel 4** · CRUD + Reporte — Rol D\n"
        )
        st.caption(f"Semilla oficial: SEED = 96 · datos: `{RUTA_DATOS.name}`")

    df = _df()
    st.success(f"Dataset limpio en memoria: {len(df):,} filas × {df.shape[1]} columnas · "
               f"{df['estacion'].nunique()} estaciones.")

    t1, t2, t3, t4 = st.tabs(
        ["🔎 1 · EDA & Clustering", "🤖 2 · Predicción", "📈 3 · Serie temporal", "🗂️ 4 · CRUD & Reporte"]
    )
    with t1:
        _panel("panel_eda", df, "Panel 1 · EDA & Clustering", "Rol A",
               "resumen del EDA + visualización de clusters (K-means)")
    with t2:
        _panel("panel_predictivo", df, "Panel 2 · Predicción de alta contaminación", "Rol B",
               "clasificación RF vs XGBoost + SHAP")
    with t3:
        _panel("panel_forecast", df, "Panel 3 · Serie temporal y pronóstico", "Rol C",
               "pronóstico ≥ 4 períodos con MAPE y RMSE")
    with t4:
        _panel("panel_crud", df, "Panel 4 · CRUD de consultas & Reporte", "Rol D",
               "CRUD de consultas + Reporte PDF")


if __name__ == "__main__":
    main()
