from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core import models


def _raiz_proyecto() -> Path:
    aqui = Path(__file__).resolve().parent
    for candidata in (aqui, *aqui.parents):
        if (candidata / "models").exists() or (candidata / "data").exists():
            return candidata
    return aqui


_RAIZ = _raiz_proyecto()

RUTA_BD = Path(os.environ.get("CRUD_DB_PATH", _RAIZ / "data" / "consultas.db"))
DIR_MODELOS = _RAIZ / "models"
NOMBRE_TABLA = "consultas"

FEATURES = ["pm_10", "so2", "no2", "o3", "co", "hora", "mes", "estacion"]
NOMBRES_MODELO = ["rf_classweight.joblib", "rf.pkl", "rf_classweight.pkl", "rf.joblib"]

ESTACIONES = [
    "ATE", "CAMPO DE MARTE", "CARABAYLLO", "HUACHIPA", "PUENTE PIEDRA",
    "SAN BORJA", "SAN JUAN DE LURIGANCHO", "SAN MARTIN DE PORRES",
    "SANTA ANITA", "VILLA MARIA DEL TRIUNFO",
]

# `tipo` distingue cómo la UI (panel_predictivo.py / panel_crud.py) debe pedir el
# dato: "numero" -> number_input con min/max/def; "categoria" -> selectbox con
# `opciones`/`def`. `estacion` no tiene min/max porque no es numérica.
CONFIG_FEATURES = {
    "pm_10": {"tipo": "numero", "etiqueta": "PM10 (ug/m3)", "min": 0.0, "max": 1000.0, "def": 80.0},
    "so2": {"tipo": "numero", "etiqueta": "SO2 (ug/m3)", "min": 0.0, "max": 500.0, "def": 15.0},
    "no2": {"tipo": "numero", "etiqueta": "NO2 (ug/m3)", "min": 0.0, "max": 500.0, "def": 35.0},
    "o3": {"tipo": "numero", "etiqueta": "O3 (ug/m3)", "min": 0.0, "max": 500.0, "def": 12.0},
    "co": {"tipo": "numero", "etiqueta": "CO (ug/m3)", "min": 0.0, "max": 20000.0, "def": 900.0},
    "hora": {"tipo": "numero", "etiqueta": "Hora del día (0-23)", "min": 0.0, "max": 23.0, "def": 12.0},
    "mes": {"tipo": "numero", "etiqueta": "Mes (1-12)", "min": 1.0, "max": 12.0, "def": 6.0},
    "estacion": {"tipo": "categoria", "etiqueta": "Estación de monitoreo", "opciones": ESTACIONES, "def": ESTACIONES[0]},
}

TIPOS_CONSULTA = ["Predicción puntual", "Reporte ciudadano", "Consulta técnica", "Otro"]

ECA_PM25 = 50.0
UMBRAL_DECISION = 0.50


# --- Capa de datos (SQLite) ------------------------------------------------

def conectar() -> sqlite3.Connection:
    RUTA_BD.parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(RUTA_BD, check_same_thread=False)
    conexion.row_factory = sqlite3.Row
    return conexion


def inicializar_bd() -> None:
    ddl = f"""
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
            hora          INTEGER,
            mes           INTEGER,
            estacion      TEXT,
            clase         INTEGER,
            etiqueta      TEXT,
            probabilidad  REAL,
            umbral        REAL,
            timestamp     TEXT    NOT NULL
        )
    """
    try:
        with conectar() as conexion:
            conexion.execute(ddl)
            _migrar_columnas_nuevas(conexion)
    except sqlite3.Error as error:
        raise RuntimeError(f"No se pudo inicializar la base de datos: {error}")


def _migrar_columnas_nuevas(conexion: sqlite3.Connection) -> None:
    existentes = {fila["name"] for fila in conexion.execute(f"PRAGMA table_info({NOMBRE_TABLA})")}
    columnas_nuevas = {"hora": "INTEGER", "mes": "INTEGER", "estacion": "TEXT"}
    for columna, tipo in columnas_nuevas.items():
        if columna not in existentes:
            conexion.execute(f"ALTER TABLE {NOMBRE_TABLA} ADD COLUMN {columna} {tipo}")


def insertar_consulta(registro: dict[str, Any]) -> int:
    columnas = (
        "nombre, correo, tipo_consulta, mensaje, "
        "pm_10, so2, no2, o3, co, hora, mes, estacion, "
        "clase, etiqueta, probabilidad, umbral, timestamp"
    )
    marcadores = ", ".join(["?"] * 17)
    valores = (
        registro["nombre"], registro["correo"], registro["tipo_consulta"], registro["mensaje"],
        registro["pm_10"], registro["so2"], registro["no2"], registro["o3"], registro["co"],
        registro["hora"], registro["mes"], registro["estacion"],
        registro["clase"], registro["etiqueta"], registro["probabilidad"], registro["umbral"],
        registro["timestamp"],
    )
    with conectar() as conexion:
        cursor = conexion.execute(
            f"INSERT INTO {NOMBRE_TABLA} ({columnas}) VALUES ({marcadores})", valores,
        )
        return int(cursor.lastrowid)


def listar_consultas() -> pd.DataFrame:
    with conectar() as conexion:
        return pd.read_sql_query(f"SELECT * FROM {NOMBRE_TABLA} ORDER BY id DESC", conexion)


def obtener_consulta(id_consulta: int) -> Optional[dict[str, Any]]:
    with conectar() as conexion:
        fila = conexion.execute(f"SELECT * FROM {NOMBRE_TABLA} WHERE id = ?", (id_consulta,)).fetchone()
    return dict(fila) if fila is not None else None


def actualizar_consulta(id_consulta: int, campos: dict[str, Any]) -> None:
    if not campos:
        return
    asignaciones = ", ".join(f"{columna} = ?" for columna in campos)
    valores = list(campos.values()) + [id_consulta]
    with conectar() as conexion:
        conexion.execute(f"UPDATE {NOMBRE_TABLA} SET {asignaciones} WHERE id = ?", valores)


def eliminar_consulta(id_consulta: int) -> None:
    with conectar() as conexion:
        conexion.execute(f"DELETE FROM {NOMBRE_TABLA} WHERE id = ?", (id_consulta,))


# --- Resolución del predictor (3 niveles, con respaldo) ----------------------

def _predecir_con_modelo(modelo, entrada: dict[str, float]) -> dict[str, Any]:
    return models.predecir_desde_entrada(modelo, entrada, umbral=UMBRAL_DECISION)


def predecir_respaldo(entrada: dict[str, float]) -> dict[str, Any]:
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


def resolver_predictor() -> dict[str, Any]:
    try:
        for nombre in NOMBRES_MODELO:
            ruta = Path(models.DIR_MODELOS) / nombre
            if ruta.exists():
                rf = models.cargar_modelo(ruta)

                def _predecir(entrada, _rf=rf):
                    return models.predecir_desde_entrada(_rf, entrada)

                return {"modo": f"real - {nombre} (vía Panel 2)", "predecir": _predecir}
    except Exception:  # noqa: BLE001 -- se intenta la siguiente opción
        pass

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

    return {"modo": "respaldo", "predecir": predecir_respaldo}
