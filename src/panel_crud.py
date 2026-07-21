"""
src/panel_crud.py — Panel 4: CRUD de consultas de predicción de calidad del aire.

Proyecto: "Dos Limas, un mismo cielo" — Dashboard de calidad del aire
(SENAMHI, Lima Metropolitana). Rol D (CRUD y reporte) — Piero Huayta.

CONTRATO DE EQUIPO (definido por Rol C en app.py):
  Este módulo vive en src/ y expone `render(df=None)`, igual que panel_eda,
  panel_predictivo y panel_forecast. app.py lo carga automáticamente vía
  importlib con `_panel("panel_crud", df, ...)`; no hay que tocar app.py.
  El parámetro `df` (dataset limpio del Rol A) se recibe por contrato aunque el
  CRUD gestione su propia persistencia en SQLite y no lo necesite.

Este módulo implementa el Panel 4 del dashboard: un CRUD completo sobre SQLite
que registra "consultas de predicción". Cada consulta almacena los datos de
entrada (concentraciones de contaminantes) y la predicción devuelta por el
modelo del Rol B (Random Forest con class_weight='balanced'), tal como lo exige
la rúbrica: «Formulario para guardar una consulta = datos de entrada +
predicción devuelta; lista de consultas guardadas; botón editar y eliminar;
timestamp automático».

Conexión con el modelo de Naze (Rol B)
--------------------------------------
`_cargar_predictor()` intenta, en este orden:
  1. Reutilizar la función oficial `predecir_desde_entrada` de `panel_predictivo`
     (contrato de CONTEXTO_ROL_B, sección 5) — es la vía preferida, no depende
     del nombre de archivo del modelo.
  2. Cargar directamente el modelo serializado con joblib, probando varios
     nombres posibles (rf_classweight.joblib, rf.pkl, rf_classweight.pkl,
     rf.joblib).
  3. Si nada existe todavía, usar un predictor de respaldo basado en reglas para
     que el CRUD siga siendo demostrable en la presentación.

Integración (automática, no requiere editar app.py)
----------------------------------------------------
    # app.py de Rol C ya hace, dentro de la pestaña 4:
    #   _panel("panel_crud", df, "Panel 4 ...", "Rol D", "...")
    # que internamente importa este módulo y llama a render(df).

Ejecución aislada para pruebas
------------------------------
    streamlit run src/panel_crud.py
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Localización robusta de la raíz del proyecto
# ---------------------------------------------------------------------------

def _raiz_proyecto() -> Path:
    """Devuelve la raíz del proyecto funcione panel_4.py en la raíz o en src/.

    Sube por el árbol de directorios buscando la carpeta que contenga `models`
    o `data`. Así las rutas a la base de datos y a los modelos son correctas sin
    importar donde se coloque este archivo.
    """
    aqui = Path(__file__).resolve().parent
    for candidata in (aqui, *aqui.parents):
        if (candidata / "models").exists() or (candidata / "data").exists():
            return candidata
    return aqui


_RAIZ = _raiz_proyecto()

# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------

# Base de datos SQLite. Se puede sobrescribir con la variable de entorno
# CRUD_DB_PATH (útil en Streamlit Cloud).
RUTA_BD = Path(os.environ.get("CRUD_DB_PATH", _RAIZ / "data" / "consultas.db"))
DIR_MODELOS = _RAIZ / "models"
DIR_SRC = _RAIZ / "src"

NOMBRE_TABLA = "consultas"

# Contaminantes usados como features del modelo del Rol B (pm_25 se excluye para
# evitar fuga de la variable objetivo). El orden es el del contrato de Rol B.
FEATURES = ["pm_10", "so2", "no2", "o3", "co"]

# Nombres candidatos del modelo Random Forest serializado por Naze.
NOMBRES_MODELO = ["rf_classweight.joblib", "rf.pkl", "rf_classweight.pkl", "rf.joblib"]

# Etiquetas legibles y valores por defecto de cada feature en el formulario.
CONFIG_FEATURES = {
    "pm_10": {"etiqueta": "PM10 (ug/m3)", "min": 0.0, "max": 1000.0, "def": 80.0},
    "so2": {"etiqueta": "SO2 (ug/m3)", "min": 0.0, "max": 500.0, "def": 15.0},
    "no2": {"etiqueta": "NO2 (ug/m3)", "min": 0.0, "max": 500.0, "def": 35.0},
    "o3": {"etiqueta": "O3 (ug/m3)", "min": 0.0, "max": 500.0, "def": 12.0},
    "co": {"etiqueta": "CO (ug/m3)", "min": 0.0, "max": 20000.0, "def": 900.0},
}

TIPOS_CONSULTA = [
    "Predicción puntual",
    "Reporte ciudadano",
    "Consulta técnica",
    "Otro",
]

# Umbral oficial del proyecto (ECA PM2.5). Solo lo usa el predictor de respaldo.
ECA_PM25 = 50.0
UMBRAL_DECISION = 0.50


# ---------------------------------------------------------------------------
# Capa de datos (SQLite)
# ---------------------------------------------------------------------------

def _conectar() -> sqlite3.Connection:
    """Abre una conexión a SQLite creando el directorio contenedor si falta."""
    RUTA_BD.parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(RUTA_BD, check_same_thread=False)
    conexion.row_factory = sqlite3.Row  # acceso a columnas por nombre
    return conexion


def inicializar_bd() -> None:
    """Crea la tabla de consultas si aún no existe (idempotente)."""
    consulta_ddl = f"""
        CREATE TABLE IF NOT EXISTS {NOMBRE_TABLA} (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre        TEXT    NOT NULL,
            correo        TEXT,
            tipo_consulta TEXT,
            mensaje       TEXT,
            pm_10         REAL,
            so2           REAL,
            no2           REAL,
            o3            REAL,
            co            REAL,
            clase         INTEGER,
            etiqueta      TEXT,
            probabilidad  REAL,
            umbral        REAL,
            timestamp     TEXT    NOT NULL
        )
    """
    try:
        with _conectar() as conexion:
            conexion.execute(consulta_ddl)
    except sqlite3.Error as error:
        raise RuntimeError(f"No se pudo inicializar la base de datos: {error}")


def insertar_consulta(registro: dict[str, Any]) -> int:
    """Inserta una nueva consulta y devuelve el id autogenerado."""
    columnas = (
        "nombre, correo, tipo_consulta, mensaje, "
        "pm_10, so2, no2, o3, co, "
        "clase, etiqueta, probabilidad, umbral, timestamp"
    )
    marcadores = ", ".join(["?"] * 14)  # 14 columnas -> 14 placeholders
    valores = (
        registro["nombre"],
        registro["correo"],
        registro["tipo_consulta"],
        registro["mensaje"],
        registro["pm_10"],
        registro["so2"],
        registro["no2"],
        registro["o3"],
        registro["co"],
        registro["clase"],
        registro["etiqueta"],
        registro["probabilidad"],
        registro["umbral"],
        registro["timestamp"],
    )
    with _conectar() as conexion:
        cursor = conexion.execute(
            f"INSERT INTO {NOMBRE_TABLA} ({columnas}) VALUES ({marcadores})",
            valores,
        )
        return int(cursor.lastrowid)


def listar_consultas() -> pd.DataFrame:
    """Devuelve todas las consultas como DataFrame, ordenadas por id descendente."""
    with _conectar() as conexion:
        return pd.read_sql_query(
            f"SELECT * FROM {NOMBRE_TABLA} ORDER BY id DESC",
            conexion,
        )


def obtener_consulta(id_consulta: int) -> Optional[dict[str, Any]]:
    """Devuelve una consulta por id como diccionario, o None si no existe."""
    with _conectar() as conexion:
        fila = conexion.execute(
            f"SELECT * FROM {NOMBRE_TABLA} WHERE id = ?",
            (id_consulta,),
        ).fetchone()
    return dict(fila) if fila is not None else None


def actualizar_consulta(id_consulta: int, campos: dict[str, Any]) -> None:
    """Actualiza los campos indicados de una consulta existente."""
    if not campos:
        return
    asignaciones = ", ".join(f"{columna} = ?" for columna in campos)
    valores = list(campos.values()) + [id_consulta]
    with _conectar() as conexion:
        conexion.execute(
            f"UPDATE {NOMBRE_TABLA} SET {asignaciones} WHERE id = ?",
            valores,
        )


def eliminar_consulta(id_consulta: int) -> None:
    """Elimina una consulta por id."""
    with _conectar() as conexion:
        conexion.execute(
            f"DELETE FROM {NOMBRE_TABLA} WHERE id = ?",
            (id_consulta,),
        )


# ---------------------------------------------------------------------------
# Integración con el modelo del Rol B (con respaldo)
# ---------------------------------------------------------------------------

def _predecir_con_modelo(modelo, entrada: dict[str, float]) -> dict[str, Any]:
    """Predice usando un estimador sklearn cargado directamente (opción 2).

    Construye un DataFrame con las columnas en el orden de FEATURES para evitar
    la advertencia de sklearn sobre nombres de features. RF/XGBoost no requieren
    escalado (CONTEXTO_ROL_B), así que se usa la entrada cruda.
    """
    X = pd.DataFrame([[entrada[f] for f in FEATURES]], columns=FEATURES)
    proba = float(modelo.predict_proba(X)[0, 1])
    clase = int(proba >= UMBRAL_DECISION)
    return {
        "clase": clase,
        "etiqueta": "Alta contaminación" if clase == 1 else "Baja contaminación",
        "probabilidad": round(proba, 4),
        "umbral": UMBRAL_DECISION,
    }


@st.cache_resource(show_spinner=False)
def _cargar_predictor() -> dict[str, Any]:
    """Carga el predictor siguiendo la estrategia de 3 niveles descrita arriba.

    Devuelve un dict con:
      - 'modo': texto descriptivo ("real ...", "respaldo", ...)
      - 'predecir': callable(entrada) -> {clase, etiqueta, probabilidad, umbral}
    """
    import sys

    if DIR_SRC.exists() and str(DIR_SRC) not in sys.path:
        sys.path.append(str(DIR_SRC))

    # --- Opción 1 (preferida): función oficial de Naze ---
    try:
        import models as modelo_rolb  # type: ignore
        from panel_predictivo import predecir_desde_entrada  # type: ignore

        dir_modelos = getattr(modelo_rolb, "DIR_MODELOS", DIR_MODELOS)
        for nombre in NOMBRES_MODELO:
            ruta = Path(dir_modelos) / nombre
            if ruta.exists():
                rf = modelo_rolb.cargar_modelo(ruta)

                def _predecir(entrada, _rf=rf):
                    return predecir_desde_entrada(_rf, entrada)

                return {"modo": f"real - {nombre} (vía Rol B)", "predecir": _predecir}
    except Exception:  # noqa: BLE001 — se intenta la siguiente opción.
        pass

    # --- Opción 2: carga directa con joblib ---
    try:
        import joblib

        for nombre in NOMBRES_MODELO:
            ruta = DIR_MODELOS / nombre
            if ruta.exists():
                modelo = joblib.load(ruta)

                def _predecir(entrada, _m=modelo):
                    return _predecir_con_modelo(_m, entrada)

                return {"modo": f"real - {nombre} (joblib)", "predecir": _predecir}
    except Exception:  # noqa: BLE001
        pass

    # --- Opción 3: respaldo heurístico ---
    return {"modo": "respaldo", "predecir": _predecir_respaldo}


def _predecir_respaldo(entrada: dict[str, float]) -> dict[str, Any]:
    """Predictor heurístico de respaldo (solo demo, NO es el modelo entrenado)."""
    import math

    pm_10 = float(entrada.get("pm_10", 0.0))
    co = float(entrada.get("co", 0.0))
    score = 0.03 * (pm_10 - 100.0) + 0.0008 * (co - 900.0)
    probabilidad = 1.0 / (1.0 + math.exp(-score))
    clase = int(probabilidad >= UMBRAL_DECISION)
    return {
        "clase": clase,
        "etiqueta": "Alta contaminación" if clase == 1 else "Baja contaminación",
        "probabilidad": round(probabilidad, 4),
        "umbral": UMBRAL_DECISION,
    }


# ---------------------------------------------------------------------------
# Secciones de interfaz (Streamlit)
# ---------------------------------------------------------------------------

def _seccion_registro(predictor: dict[str, Any]) -> None:
    """Formulario de creación (Create): captura datos, predice y persiste."""
    st.subheader(":material/edit_note: Registrar nueva consulta")
    st.caption(f"Predictor activo: **{predictor['modo']}**")

    if predictor["modo"] == "respaldo":
        st.info(
            "Modelo del Rol B no encontrado: se usa un predictor de respaldo para "
            "la demo. Genera el modelo real con `uv run python src/models.py`.",
            icon=":material/warning:",
        )

    with st.container(border=True), st.form("form_registro", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            nombre = st.text_input("Nombre *", max_chars=120)
            correo = st.text_input("Correo", max_chars=120)
        with col_b:
            tipo_consulta = st.selectbox("Tipo de consulta", TIPOS_CONSULTA)
            mensaje = st.text_area("Mensaje / observación", height=80)

        st.markdown("**Datos de entrada del modelo (contaminantes)**")
        columnas = st.columns(len(FEATURES))
        entrada: dict[str, float] = {}
        for columna, feature in zip(columnas, FEATURES):
            cfg = CONFIG_FEATURES[feature]
            with columna:
                entrada[feature] = st.number_input(
                    cfg["etiqueta"],
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["def"],
                    step=1.0,
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
            nuevo_id = insertar_consulta(registro)
            st.session_state["ultima_prediccion"] = prediccion
            st.success(
                f"Consulta #{nuevo_id} guardada. "
                f"Predicción: **{registro['etiqueta']}** "
                f"(prob. {registro['probabilidad']:.2f})."
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"No se pudo guardar la consulta: {error}")


def _seccion_listado() -> None:
    """Listado de consultas (Read) en una tabla interactiva."""
    st.subheader(":material/list_alt: Consultas registradas")
    df = listar_consultas()

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
    """Edición de una consulta existente (Update)."""
    st.subheader(":material/edit: Editar consulta")
    df = listar_consultas()

    if df.empty:
        st.caption("No hay registros para editar.")
        return

    id_sel = st.selectbox("Selecciona el id a editar", options=df["id"].tolist(), key="edit_id")
    registro = obtener_consulta(int(id_sel))
    if registro is None:
        st.warning("El registro seleccionado ya no existe.")
        return

    with st.container(border=True), st.form("form_edicion"):
        col_a, col_b = st.columns(2)
        with col_a:
            nombre = st.text_input("Nombre", value=registro["nombre"] or "")
            correo = st.text_input("Correo", value=registro["correo"] or "")
        with col_b:
            indice_tipo = (
                TIPOS_CONSULTA.index(registro["tipo_consulta"])
                if registro["tipo_consulta"] in TIPOS_CONSULTA
                else 0
            )
            tipo_consulta = st.selectbox("Tipo de consulta", TIPOS_CONSULTA, index=indice_tipo)
            mensaje = st.text_area("Mensaje", value=registro["mensaje"] or "")

        guardar = st.form_submit_button("Guardar cambios", type="primary")

    if guardar:
        try:
            actualizar_consulta(
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
    """Eliminación de una consulta (Delete) con confirmación explícita."""
    st.subheader(":material/delete: Eliminar consulta")
    df = listar_consultas()

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
                eliminar_consulta(int(id_sel))
                st.success(f"Consulta #{id_sel} eliminada.")
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(f"No se pudo eliminar: {error}")


# ---------------------------------------------------------------------------
# Punto de entrada del panel
# ---------------------------------------------------------------------------

def render(df=None) -> None:
    """Renderiza el Panel 4 completo (CRUD).

    Firma según el contrato del equipo: `render(df=None)`. El parámetro `df`
    (dataset limpio del Rol A) se acepta por compatibilidad con app.py; este
    panel no lo necesita porque administra su propia persistencia en SQLite.
    """
    st.header(":material/folder_open: Panel 4 — CRUD de consultas y predicción")
    st.caption(
        "Registra consultas con los datos de entrada del modelo, obtén la "
        "predicción del Rol B y administra el historial (crear, leer, editar, "
        "eliminar). Persistencia local en SQLite."
    )

    try:
        inicializar_bd()
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