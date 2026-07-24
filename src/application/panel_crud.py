from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from core import consultas as C

FEATURES = C.FEATURES
CONFIG_FEATURES = C.CONFIG_FEATURES
TIPOS_CONSULTA = C.TIPOS_CONSULTA


@st.cache_resource(show_spinner=False)
def _cargar_predictor() -> dict:
    return C.resolver_predictor()


# ---------------------------------------------------------------------------
# Secciones de interfaz (Streamlit)
# ---------------------------------------------------------------------------

def _seccion_registro(predictor: dict) -> None:
    st.subheader(":material/edit_note: Registrar nueva consulta")
    st.caption(
        "Completa los datos de contacto y las lecturas de contaminantes: el modelo "
        "genera una predicción y la consulta queda guardada en el historial."
    )
    st.caption(f"Predictor activo: **{predictor['modo']}**")

    if predictor["modo"] == "respaldo":
        st.info(
            "Modelo del Panel 2 no encontrado: se usa un predictor de respaldo para "
            "la demo. Genera el modelo real con `uv run python src/core/models.py`.",
            icon=":material/warning:",
        )

    with st.container(border=True), st.form("form_registro", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            nombre = st.text_input("Nombre *", max_chars=120)
            correo = st.text_input("Correo", max_chars=120)
        with col_b:
            tipo_consulta = st.segmented_control("Tipo de consulta", TIPOS_CONSULTA,
                                                  default=TIPOS_CONSULTA[0])
            mensaje = st.text_area("Mensaje / observación", height=80)

        st.markdown("**Datos de entrada del modelo (contaminantes, hora, mes, estación)**")
        columnas = st.columns(len(FEATURES))
        entrada: dict[str, float | str] = {}
        for columna, feature in zip(columnas, FEATURES):
            cfg = CONFIG_FEATURES[feature]
            with columna:
                if cfg["tipo"] == "categoria":
                    entrada[feature] = st.selectbox(
                        cfg["etiqueta"], options=cfg["opciones"],
                        index=cfg["opciones"].index(cfg["def"]),
                    )
                else:
                    entrada[feature] = st.number_input(
                        cfg["etiqueta"], min_value=cfg["min"], max_value=cfg["max"],
                        value=cfg["def"], step=1.0,
                    )

        enviado = st.form_submit_button("Predecir y guardar", type="primary")

    if enviado:
        if not nombre.strip():
            st.error("El campo **Nombre** es obligatorio.")
            return
        try:
            prediccion = predictor["predecir"](entrada)
            registro = {
                "nombre": nombre.strip(),
                "correo": correo.strip(),
                "tipo_consulta": tipo_consulta,
                "mensaje": mensaje.strip(),
                **entrada,
                "clase": prediccion.get("clase"),
                "etiqueta": prediccion.get("etiqueta"),
                "probabilidad": prediccion.get("probabilidad"),
                "umbral": prediccion.get("umbral"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            nuevo_id = C.insertar_consulta(registro)
            st.session_state["ultima_prediccion"] = prediccion
            st.success(
                f"Consulta #{nuevo_id} guardada. "
                f"Predicción: **{registro['etiqueta']}** "
                f"(prob. {registro['probabilidad']:.2f})."
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"No se pudo guardar la consulta: {error}")


def _seccion_listado() -> None:
    st.subheader(":material/list_alt: Consultas registradas")
    st.caption("Historial completo: datos de contacto, lecturas ingresadas y predicción obtenida.")
    df = C.listar_consultas()

    if df.empty:
        st.caption("Aún no hay consultas registradas.")
        return

    with st.container(border=True):
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            ":material/download: Descargar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="consultas.csv",
            mime="text/csv",
        )


def _seccion_edicion() -> None:
    st.subheader(":material/edit: Editar consulta")
    st.caption(
        "Corrige los datos de contacto o el motivo de una consulta ya guardada. "
        "No vuelve a ejecutar el modelo: la predicción original se conserva."
    )
    df = C.listar_consultas()

    if df.empty:
        st.caption("No hay registros para editar.")
        return

    id_sel = st.selectbox("Selecciona el id a editar", options=df["id"].tolist(), key="edit_id")
    registro = C.obtener_consulta(int(id_sel))
    if registro is None:
        st.warning("El registro seleccionado ya no existe.")
        return

    with st.container(border=True), st.form("form_edicion"):
        col_a, col_b = st.columns(2)
        with col_a:
            nombre = st.text_input("Nombre", value=registro["nombre"] or "")
            correo = st.text_input("Correo", value=registro["correo"] or "")
        with col_b:
            tipo_actual = (
                registro["tipo_consulta"] if registro["tipo_consulta"] in TIPOS_CONSULTA
                else TIPOS_CONSULTA[0]
            )
            tipo_consulta = st.segmented_control("Tipo de consulta", TIPOS_CONSULTA,
                                                  default=tipo_actual)
            mensaje = st.text_area("Mensaje", value=registro["mensaje"] or "")

        guardar = st.form_submit_button("Guardar cambios", type="primary")

    if guardar:
        try:
            C.actualizar_consulta(
                int(id_sel),
                {
                    "nombre": nombre.strip(),
                    "correo": correo.strip(),
                    "tipo_consulta": tipo_consulta,
                    "mensaje": mensaje.strip(),
                },
            )
            st.success(f"Consulta #{id_sel} actualizada.")
            st.rerun()
        except Exception as error:  # noqa: BLE001
            st.error(f"No se pudo actualizar: {error}")


def _seccion_eliminacion() -> None:
    st.subheader(":material/delete: Eliminar consulta")
    st.caption("Elimina permanentemente una consulta del historial. Esta acción no se puede deshacer.")
    df = C.listar_consultas()

    if df.empty:
        st.caption("No hay registros para eliminar.")
        return

    with st.container(border=True):
        id_sel = st.selectbox("Selecciona el id a eliminar", options=df["id"].tolist(), key="delete_id")
        confirmar = st.checkbox(
            f"Confirmo que deseo eliminar la consulta #{id_sel} (acción irreversible)."
        )

        if st.button("Eliminar", type="secondary", disabled=not confirmar):
            try:
                C.eliminar_consulta(int(id_sel))
                st.success(f"Consulta #{id_sel} eliminada.")
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(f"No se pudo eliminar: {error}")


# ---------------------------------------------------------------------------
# Punto de entrada del panel
# ---------------------------------------------------------------------------

def render(df=None) -> None:
    st.header(":material/folder_open: Panel 4 — CRUD de consultas y predicción")
    st.caption(
        "Registra consultas con los datos de entrada del modelo, obtén la "
        "predicción del modelo del Panel 2 y administra el historial (crear, "
        "leer, editar, eliminar). Persistencia local en SQLite."
    )

    try:
        C.inicializar_bd()
    except RuntimeError as error:
        st.error(str(error))
        return

    predictor = _cargar_predictor()

    tab_crear, tab_listar, tab_editar, tab_eliminar = st.tabs(
        [":material/edit_note: Crear", ":material/list_alt: Listar", ":material/edit: Editar", ":material/delete: Eliminar"]
    )
    with tab_crear:
        _seccion_registro(predictor)
    with tab_listar:
        _seccion_listado()
    with tab_editar:
        _seccion_edicion()
    with tab_eliminar:
        _seccion_eliminacion()


if __name__ == "__main__":
    st.set_page_config(page_title="Panel 4 — CRUD", layout="wide")
    render()
