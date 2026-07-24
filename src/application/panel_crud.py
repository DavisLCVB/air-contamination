from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from core import consultas as C

# theme.py vive junto a los paneles (src/application). Se importa de forma
# defensiva: si no está disponible, los gráficos usan el estilo Altair por defecto.
try:
    from application import theme  # type: ignore
except ImportError:
    try:
        import theme  # type: ignore
    except ImportError:
        theme = None  # type: ignore[assignment]

FEATURES = C.FEATURES
CONTAMINANTES = C.CONTAMINANTES
CONFIG_FEATURES = C.CONFIG_FEATURES
TIPOS_CONSULTA = C.TIPOS_CONSULTA

# Orden de columnas para la tabla. Se toma del backend si está disponible;
# si `core.consultas` es una versión anterior sin COLUMNAS_CANONICAS, se usa
# este respaldo local para que el panel nunca se caiga por un AttributeError.
_ORDEN_RESPALDO = [
    "id", "nombre", "correo", "tipo_consulta", "mensaje",
    "pm_10", "so2", "no2", "o3", "co", "hora", "mes", "estacion",
    "clase", "etiqueta", "probabilidad", "umbral", "timestamp",
]
COLUMNAS_CANONICAS = getattr(C, "COLUMNAS_CANONICAS", _ORDEN_RESPALDO)

# Colores de respaldo (Catppuccin Mocha) si theme.py no expone la paleta.
_COLOR_ALTA = "#f38ba8"   # red
_COLOR_BAJA = "#a6e3a1"   # green
_TEXTO_BADGE = "#1e1e2e"  # base


@st.cache_resource(show_spinner=False)
def _cargar_predictor() -> dict:
    return C.resolver_predictor()


def _aplicar_tema_altair() -> None:
    """Registra el tema Altair del proyecto (Mocha/Latte) si está disponible."""
    if theme is not None and hasattr(theme, "aplicar_estilo_altair"):
        try:
            theme.aplicar_estilo_altair()
        except Exception:  # noqa: BLE001 -- el estilo nunca debe romper el panel
            pass


def _badge_prediccion(etiqueta: str, probabilidad: float | None, clase: int | None) -> None:
    """Badge HTML rojo/verde según la clase predicha, con la paleta del proyecto."""
    color = _COLOR_ALTA if clase == 1 else _COLOR_BAJA
    if theme is not None:
        paleta = getattr(theme, "PALETA", None) or getattr(theme, "COLORES", None)
        if isinstance(paleta, dict):
            color = paleta.get("red" if clase == 1 else "green", color)
    prob_txt = f" · prob. {probabilidad:.2f}" if probabilidad is not None else ""
    st.markdown(
        f"""
        <span style="
            background:{color}; color:{_TEXTO_BADGE};
            padding:0.25rem 0.75rem; border-radius:999px;
            font-weight:700; font-size:0.9rem; display:inline-block;">
            {etiqueta}{prob_txt}
        </span>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pestaña 1 — Registrar consulta
# ---------------------------------------------------------------------------

def _tab_registro(predictor: dict) -> None:
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
            correo = st.text_input("Correo", max_chars=120,
                                   placeholder="opcional — usuario@dominio.tld")
        with col_b:
            tipo_consulta = st.segmented_control("Tipo de consulta", TIPOS_CONSULTA,
                                                  default=TIPOS_CONSULTA[0])
            mensaje = st.text_area("Mensaje / observación", height=80)

        st.markdown("**Lecturas de contaminantes (µg/m³)**")
        entrada: dict[str, float | str] = {}
        for columna, feature in zip(st.columns(len(CONTAMINANTES)), CONTAMINANTES):
            cfg = CONFIG_FEATURES[feature]
            with columna:
                entrada[feature] = st.number_input(
                    cfg["etiqueta"], min_value=cfg["min"], max_value=cfg["max"],
                    value=cfg["def"], step=1.0,
                )

        st.markdown("**Contexto temporal y espacial**")
        col_hora, col_mes, col_estacion = st.columns([1, 1, 2])
        with col_hora:
            entrada["hora"] = st.number_input(
                CONFIG_FEATURES["hora"]["etiqueta"], min_value=0.0, max_value=23.0,
                value=CONFIG_FEATURES["hora"]["def"], step=1.0,
            )
        with col_mes:
            entrada["mes"] = st.number_input(
                CONFIG_FEATURES["mes"]["etiqueta"], min_value=1.0, max_value=12.0,
                value=CONFIG_FEATURES["mes"]["def"], step=1.0,
            )
        with col_estacion:
            cfg_est = CONFIG_FEATURES["estacion"]
            entrada["estacion"] = st.selectbox(
                cfg_est["etiqueta"], options=cfg_est["opciones"],
                index=cfg_est["opciones"].index(cfg_est["def"]),
            )

        enviado = st.form_submit_button("Predecir y guardar", type="primary")

    if not enviado:
        return

    # Validación en la UI (el backend vuelve a validar de todos modos).
    if not nombre.strip():
        st.error("El campo **Nombre** es obligatorio.")
        return
    if not C.validar_correo(correo):
        st.error("El **correo** no tiene un formato válido (usuario@dominio.tld).")
        return

    try:
        prediccion = predictor["predecir"](entrada)
        nuevo_id = C.guardar_consulta(
            nombre=nombre, correo=correo, tipo_consulta=tipo_consulta,
            mensaje=mensaje, contaminantes=entrada, prediccion=prediccion,
        )
        st.session_state["ultima_prediccion"] = prediccion
        _badge_prediccion(
            prediccion.get("etiqueta", "—"),
            prediccion.get("probabilidad"),
            prediccion.get("clase"),
        )
        st.success(f"Consulta #{nuevo_id} guardada en el historial.")
        st.toast(f"Consulta #{nuevo_id}: {prediccion.get('etiqueta', '—')}",
                 icon=":material/check_circle:")
    except (ValueError, RuntimeError) as error:
        st.error(f"No se pudo guardar la consulta: {error}")
    except Exception as error:  # noqa: BLE001
        st.error(f"Error inesperado al predecir o guardar: {error}")


# ---------------------------------------------------------------------------
# Pestaña 2 — Listar y gestionar (filtros + editar + eliminar)
# ---------------------------------------------------------------------------

def _tab_gestion() -> None:
    st.subheader(":material/list_alt: Listar y gestionar consultas")
    st.caption("Busca en el historial, edita datos de contacto o elimina registros.")

    col_busqueda, col_tipo = st.columns([2, 1])
    with col_busqueda:
        busqueda = st.text_input(
            "Buscar por nombre o correo", placeholder="Búsqueda parcial…",
            key="crud_busqueda",
        )
    with col_tipo:
        tipo_sel = st.selectbox(
            "Tipo de consulta", options=["Todos", *TIPOS_CONSULTA], key="crud_filtro_tipo",
        )

    try:
        df = C.obtener_consultas(
            filtro_tipo=None if tipo_sel == "Todos" else tipo_sel,
            busqueda=busqueda,
        )
    except RuntimeError as error:
        st.error(str(error))
        return

    if df.empty:
        st.caption("No hay consultas que coincidan con los filtros.")
        return

    # Orden de columnas consistente con el esquema canónico del backend
    # (cualquier columna inesperada queda al final en vez de romper el panel).
    orden = [c for c in COLUMNAS_CANONICAS if c in df.columns]
    df = df[orden + [c for c in df.columns if c not in orden]]

    with st.container(border=True):
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "nombre": st.column_config.TextColumn("Nombre", width="medium"),
                "correo": st.column_config.TextColumn("Correo", width="medium"),
                "tipo_consulta": st.column_config.TextColumn("Tipo", width="small"),
                "mensaje": st.column_config.TextColumn("Mensaje", width="large"),
                "probabilidad": st.column_config.ProgressColumn(
                    "Prob. alta", format="%.2f", min_value=0.0, max_value=1.0,
                ),
                "etiqueta": st.column_config.TextColumn("Predicción", width="medium"),
                "timestamp": st.column_config.TextColumn("Fecha", width="medium"),
            },
        )
        st.caption(f"{len(df)} registro(s) mostrados.")

    ids_disponibles = df["id"].tolist()

    with st.expander(":material/edit: Editar una consulta"):
        _subformulario_edicion(ids_disponibles)

    with st.expander(":material/delete: Eliminar una consulta"):
        _subformulario_eliminacion(ids_disponibles)


def _subformulario_edicion(ids_disponibles: list[int]) -> None:
    st.caption(
        "Corrige los datos de contacto o el motivo de una consulta ya guardada. "
        "No vuelve a ejecutar el modelo: la predicción original se conserva."
    )
    id_sel = st.selectbox("Selecciona el id a editar", options=ids_disponibles, key="edit_id")
    registro = C.obtener_consulta(int(id_sel))
    if registro is None:
        st.warning("El registro seleccionado ya no existe.")
        return

    with st.form("form_edicion"):
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
        if not nombre.strip():
            st.error("El campo **Nombre** es obligatorio.")
            return
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
        except (ValueError, RuntimeError) as error:
            st.error(f"No se pudo actualizar: {error}")


def _subformulario_eliminacion(ids_disponibles: list[int]) -> None:
    st.caption("Elimina permanentemente una consulta del historial. Esta acción no se puede deshacer.")
    id_sel = st.selectbox("Selecciona el id a eliminar", options=ids_disponibles, key="delete_id")
    confirmar = st.checkbox(
        f"Confirmo que deseo eliminar la consulta #{id_sel} (acción irreversible)."
    )
    if st.button("Eliminar", type="secondary", disabled=not confirmar):
        try:
            C.eliminar_consulta(int(id_sel))
            st.success(f"Consulta #{id_sel} eliminada.")
            st.rerun()
        except (ValueError, RuntimeError) as error:
            st.error(f"No se pudo eliminar: {error}")


# ---------------------------------------------------------------------------
# Pestaña 3 — Reportes y analítica
# ---------------------------------------------------------------------------

def _tab_reportes() -> None:
    st.subheader(":material/monitoring: Reportes y analítica del historial")
    st.caption("Métricas agregadas de las consultas registradas en este panel.")

    try:
        resumen = C.obtener_resumen_estadistico()
    except RuntimeError as error:
        st.error(str(error))
        return

    if resumen["total"] == 0:
        st.caption("Aún no hay consultas registradas: la analítica aparecerá aquí.")
        return

    # --- Tarjetas de métricas ---
    promedios = resumen["promedios_contaminantes"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de consultas", resumen["total"])
    col2.metric("Tipo más frecuente", resumen["tipo_mas_frecuente"] or "—")
    col3.metric("Promedio PM10 (µg/m³)",
                f"{promedios.get('pm_10', float('nan')):.1f}" if "pm_10" in promedios else "—")
    col4.metric("Prob. media de alta",
                f"{resumen['probabilidad_media']:.2f}" if resumen["probabilidad_media"] is not None else "—")

    # --- Gráficos Altair (paleta Mocha/Latte del proyecto) ---
    col_dona, col_linea = st.columns(2)

    with col_dona, st.container(border=True):
        st.markdown("**Distribución por tipo de consulta**")
        df_tipo = pd.DataFrame(
            {"tipo": list(resumen["por_tipo"].keys()),
             "cantidad": list(resumen["por_tipo"].values())}
        )
        dona = (
            alt.Chart(df_tipo)
            .mark_arc(innerRadius=55, cornerRadius=4)
            .encode(
                theta=alt.Theta("cantidad:Q"),
                color=alt.Color("tipo:N", legend=alt.Legend(title="Tipo")),
                tooltip=[alt.Tooltip("tipo:N", title="Tipo"),
                         alt.Tooltip("cantidad:Q", title="Consultas")],
            )
            .properties(height=260)
        )
        st.altair_chart(dona, use_container_width=True)

    with col_linea, st.container(border=True):
        st.markdown("**Evolución diaria de consultas**")
        df_dia = resumen["consultas_por_dia"].copy()
        df_dia["fecha"] = pd.to_datetime(df_dia["fecha"])
        evolucion = (
            alt.Chart(df_dia)
            .mark_area(line=True, opacity=0.45, interpolate="monotone")
            .encode(
                x=alt.X("fecha:T", title="Fecha"),
                y=alt.Y("cantidad:Q", title="Consultas"),
                tooltip=[alt.Tooltip("fecha:T", title="Fecha"),
                         alt.Tooltip("cantidad:Q", title="Consultas")],
            )
            .properties(height=260)
        )
        st.altair_chart(evolucion, use_container_width=True)

    with st.container(border=True):
        st.markdown("**Promedio de contaminantes ingresados**")
        df_prom = pd.DataFrame(
            {"contaminante": [c.upper().replace("_", "") for c in promedios],
             "promedio": list(promedios.values())}
        )
        barras = (
            alt.Chart(df_prom)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("contaminante:N", sort="-x", title=None),
                x=alt.X("promedio:Q", title="Promedio (µg/m³)"),
                color=alt.Color("contaminante:N", legend=None),
                tooltip=[alt.Tooltip("contaminante:N", title="Contaminante"),
                         alt.Tooltip("promedio:Q", title="Promedio", format=".2f")],
            )
            .properties(height=220)
        )
        st.altair_chart(barras, use_container_width=True)

    # --- Descargas ---
    _seccion_descargas()


def _df_a_excel(df: pd.DataFrame) -> bytes | None:
    """Serializa el historial a XLSX; devuelve None si falta openpyxl."""
    try:
        import openpyxl  # noqa: F401 -- solo se verifica la disponibilidad
    except ImportError:
        return None
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="consultas")
    return buffer.getvalue()


def _seccion_descargas() -> None:
    try:
        df = C.obtener_consultas()
    except RuntimeError as error:
        st.error(str(error))
        return
    if df.empty:
        return

    st.markdown("**Descargar historial completo**")
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            ":material/download: Descargar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="consultas.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_xlsx:
        excel = _df_a_excel(df)
        if excel is not None:
            st.download_button(
                ":material/download: Descargar Excel",
                data=excel,
                file_name="consultas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        else:
            st.caption("Instala `openpyxl` para habilitar la descarga en Excel.")


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

    _aplicar_tema_altair()
    predictor = _cargar_predictor()

    tab_registrar, tab_gestionar, tab_reportes = st.tabs(
        [
            ":material/edit_note: Registrar Consulta",
            ":material/list_alt: Listar y Gestionar",
            ":material/monitoring: Reportes y Analítica",
        ]
    )
    with tab_registrar:
        _tab_registro(predictor)
    with tab_gestionar:
        _tab_gestion()
    with tab_reportes:
        _tab_reportes()


if __name__ == "__main__":
    st.set_page_config(page_title="Panel 4 — CRUD", layout="wide")
    render()