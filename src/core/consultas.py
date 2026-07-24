from __future__ import annotations

import os
import re
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
CONTAMINANTES = ["pm_10", "so2", "no2", "o3", "co"]
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

# Esquema canónico de la tabla (nombre -> tipo SQLite). El orden importa: es el
# que usa la reconstrucción de la tabla y el orden de columnas en la UI.
ESQUEMA_CANONICO: dict[str, str] = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "nombre": "TEXT NOT NULL",
    "correo": "TEXT",
    "tipo_consulta": "TEXT",
    "mensaje": "TEXT",
    "pm_10": "REAL",
    "so2": "REAL",
    "no2": "REAL",
    "o3": "REAL",
    "co": "REAL",
    "hora": "INTEGER",
    "mes": "INTEGER",
    "estacion": "TEXT",
    "clase": "INTEGER",
    "etiqueta": "TEXT",
    "probabilidad": "REAL",
    "umbral": "REAL",
    "timestamp": "TEXT NOT NULL",
}
COLUMNAS_CANONICAS = list(ESQUEMA_CANONICO)

# Columnas de versiones anteriores de la BD -> su equivalente canónico. La
# migración copia el dato legado donde la columna canónica esté vacía y luego
# reconstruye la tabla solo con el esquema canónico (elimina las duplicadas).
MAPA_COLUMNAS_LEGADAS: dict[str, str] = {
    "pm10": "pm_10",
    "fecha_registro": "timestamp",
    "prediccion_pm25": "probabilidad",
    "categoria_calidad": "etiqueta",
}

# Columnas editables desde la UI (lista blanca para UPDATE dinámico seguro).
COLUMNAS_EDITABLES = {
    "nombre", "correo", "tipo_consulta", "mensaje",
    "pm_10", "so2", "no2", "o3", "co", "hora", "mes", "estacion",
}

# Validación pragmática de correo: algo@dominio.tld (no pretende ser RFC 5322).
_PATRON_CORREO = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def validar_correo(correo: str) -> bool:
    """True si el correo está vacío (campo opcional) o tiene formato válido."""
    correo = (correo or "").strip()
    return correo == "" or bool(_PATRON_CORREO.match(correo))


# --- Capa de datos (SQLite) ------------------------------------------------

def conectar() -> sqlite3.Connection:
    RUTA_BD.parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(RUTA_BD, check_same_thread=False)
    conexion.row_factory = sqlite3.Row
    return conexion


def _ddl(nombre_tabla: str, si_no_existe: bool = True) -> str:
    columnas = ",\n            ".join(
        f"{columna} {tipo}" for columna, tipo in ESQUEMA_CANONICO.items()
    )
    clausula = "IF NOT EXISTS " if si_no_existe else ""
    return f"CREATE TABLE {clausula}{nombre_tabla} (\n            {columnas}\n        )"


def inicializar_bd() -> None:
    try:
        with conectar() as conexion:
            conexion.execute(_ddl(NOMBRE_TABLA))
            _migrar_columnas_nuevas(conexion)
            _consolidar_esquema_legado(conexion)
    except sqlite3.Error as error:
        raise RuntimeError(f"No se pudo inicializar la base de datos: {error}")


# Alias de compatibilidad pedido por el equipo.
init_db = inicializar_bd


def _columnas_existentes(conexion: sqlite3.Connection) -> set[str]:
    return {fila["name"] for fila in conexion.execute(f"PRAGMA table_info({NOMBRE_TABLA})")}


def _migrar_columnas_nuevas(conexion: sqlite3.Connection) -> None:
    """Añade cualquier columna canónica que falte en una BD antigua.

    ALTER TABLE no admite NOT NULL sin default, así que las columnas se añaden
    sin restricción; la reconstrucción posterior restituye el esquema exacto.
    """
    existentes = _columnas_existentes(conexion)
    for columna, tipo in ESQUEMA_CANONICO.items():
        if columna not in existentes:
            tipo_base = tipo.replace(" NOT NULL", "")
            conexion.execute(f"ALTER TABLE {NOMBRE_TABLA} ADD COLUMN {columna} {tipo_base}")


def _consolidar_esquema_legado(conexion: sqlite3.Connection) -> None:
    """Migra columnas de versiones antiguas de la BD al esquema canónico.

    Versiones previas guardaban `pm10`, `fecha_registro`, `prediccion_pm25` y
    `categoria_calidad`. Esta migración (idempotente, corre una sola vez):
      1. Copia el valor legado a la columna canónica donde esta esté NULL.
      2. Deriva `clase` y `umbral` faltantes a partir de la etiqueta.
      3. Reconstruye la tabla solo con las columnas canónicas, eliminando las
         duplicadas (patrón CREATE + INSERT SELECT + DROP + RENAME, compatible
         con cualquier versión de SQLite).
    """
    existentes = _columnas_existentes(conexion)
    legadas = [col for col in MAPA_COLUMNAS_LEGADAS if col in existentes]
    if not legadas:
        return

    # 1) Volcar datos legados en las columnas canónicas vacías.
    for legada in legadas:
        canonica = MAPA_COLUMNAS_LEGADAS[legada]
        conexion.execute(
            f"UPDATE {NOMBRE_TABLA} SET {canonica} = COALESCE({canonica}, {legada})"
        )

    # 2) Completar clase/umbral en registros antiguos que solo tenían etiqueta.
    conexion.execute(
        f"""UPDATE {NOMBRE_TABLA}
            SET clase = CASE WHEN etiqueta LIKE 'Alta%' THEN 1 ELSE 0 END
            WHERE clase IS NULL AND etiqueta IS NOT NULL"""
    )
    conexion.execute(
        f"UPDATE {NOMBRE_TABLA} SET umbral = ? WHERE umbral IS NULL AND probabilidad IS NOT NULL",
        (UMBRAL_DECISION,),
    )

    # 3) Reconstruir la tabla con el esquema canónico exacto.
    tabla_temporal = f"{NOMBRE_TABLA}_migracion"
    conexion.execute(f"DROP TABLE IF EXISTS {tabla_temporal}")
    conexion.execute(_ddl(tabla_temporal, si_no_existe=False))
    columnas = ", ".join(COLUMNAS_CANONICAS)
    seleccion = ", ".join(
        "COALESCE(timestamp, '')" if col == "timestamp"
        else "COALESCE(nombre, '')" if col == "nombre"
        else col
        for col in COLUMNAS_CANONICAS
    )
    conexion.execute(
        f"INSERT INTO {tabla_temporal} ({columnas}) SELECT {seleccion} FROM {NOMBRE_TABLA}"
    )
    conexion.execute(f"DROP TABLE {NOMBRE_TABLA}")
    conexion.execute(f"ALTER TABLE {tabla_temporal} RENAME TO {NOMBRE_TABLA}")


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
    try:
        with conectar() as conexion:
            cursor = conexion.execute(
                f"INSERT INTO {NOMBRE_TABLA} ({columnas}) VALUES ({marcadores})", valores,
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as error:
        raise RuntimeError(f"No se pudo insertar la consulta: {error}")


def guardar_consulta(
    nombre: str,
    correo: str,
    tipo_consulta: str,
    mensaje: str,
    contaminantes: dict[str, Any],
    prediccion: dict[str, Any],
) -> int:
    """Valida los datos de contacto y persiste el registro completo.

    `contaminantes` acepta las claves de FEATURES (pm_10..co, hora, mes,
    estacion); las que falten se guardan como NULL. `prediccion` es el dict
    devuelto por el predictor activo (clase/etiqueta/probabilidad/umbral).
    """
    from datetime import datetime

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValueError("El campo 'Nombre' es obligatorio.")

    correo = (correo or "").strip()
    if not validar_correo(correo):
        raise ValueError(f"El correo '{correo}' no tiene un formato válido (usuario@dominio.tld).")

    registro = {
        "nombre": nombre,
        "correo": correo,
        "tipo_consulta": tipo_consulta or TIPOS_CONSULTA[0],
        "mensaje": (mensaje or "").strip(),
        "pm_10": contaminantes.get("pm_10"),
        "so2": contaminantes.get("so2"),
        "no2": contaminantes.get("no2"),
        "o3": contaminantes.get("o3"),
        "co": contaminantes.get("co"),
        "hora": contaminantes.get("hora"),
        "mes": contaminantes.get("mes"),
        "estacion": contaminantes.get("estacion"),
        "clase": prediccion.get("clase"),
        "etiqueta": prediccion.get("etiqueta"),
        "probabilidad": prediccion.get("probabilidad"),
        "umbral": prediccion.get("umbral"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return insertar_consulta(registro)


def obtener_consultas(
    filtro_tipo: Optional[str] = None,
    busqueda: Optional[str] = None,
    orden: str = "id",
) -> pd.DataFrame:
    """Lista consultas con filtro exacto por tipo y búsqueda LIKE en nombre/correo.

    `orden` acepta "id" o "timestamp" (siempre descendente). Los parámetros van
    parametrizados (?) — nunca interpolados — para evitar inyección SQL.
    """
    columna_orden = "timestamp" if orden == "timestamp" else "id"
    sql = f"SELECT * FROM {NOMBRE_TABLA}"
    condiciones: list[str] = []
    parametros: list[Any] = []

    if filtro_tipo:
        condiciones.append("tipo_consulta = ?")
        parametros.append(filtro_tipo)

    busqueda = (busqueda or "").strip()
    if busqueda:
        patron = f"%{busqueda}%"
        condiciones.append("(nombre LIKE ? OR correo LIKE ?)")
        parametros.extend([patron, patron])

    if condiciones:
        sql += " WHERE " + " AND ".join(condiciones)
    # `id DESC` como desempate: varios registros pueden compartir timestamp.
    sql += f" ORDER BY {columna_orden} DESC, id DESC"

    try:
        with conectar() as conexion:
            return pd.read_sql_query(sql, conexion, params=parametros)
    except sqlite3.Error as error:
        raise RuntimeError(f"No se pudo listar las consultas: {error}")


def listar_consultas() -> pd.DataFrame:
    """Compatibilidad con la firma previa: historial completo sin filtros."""
    return obtener_consultas()


def obtener_consulta(id_consulta: int) -> Optional[dict[str, Any]]:
    try:
        with conectar() as conexion:
            fila = conexion.execute(
                f"SELECT * FROM {NOMBRE_TABLA} WHERE id = ?", (id_consulta,)
            ).fetchone()
    except sqlite3.Error as error:
        raise RuntimeError(f"No se pudo leer la consulta #{id_consulta}: {error}")
    return dict(fila) if fila is not None else None


def actualizar_consulta(id_consulta: int, campos: dict[str, Any]) -> None:
    """Edición segura: valida existencia del ID y filtra columnas permitidas."""
    campos = {k: v for k, v in campos.items() if k in COLUMNAS_EDITABLES}
    if not campos:
        return
    if obtener_consulta(id_consulta) is None:
        raise ValueError(f"La consulta #{id_consulta} no existe.")
    if "correo" in campos and not validar_correo(str(campos["correo"] or "")):
        raise ValueError("El correo no tiene un formato válido.")

    asignaciones = ", ".join(f"{columna} = ?" for columna in campos)
    valores = list(campos.values()) + [id_consulta]
    try:
        with conectar() as conexion:
            conexion.execute(
                f"UPDATE {NOMBRE_TABLA} SET {asignaciones} WHERE id = ?", valores
            )
    except sqlite3.Error as error:
        raise RuntimeError(f"No se pudo actualizar la consulta #{id_consulta}: {error}")


def eliminar_consulta(id_consulta: int) -> None:
    """Borrado seguro: valida primero que el registro exista."""
    if obtener_consulta(id_consulta) is None:
        raise ValueError(f"La consulta #{id_consulta} no existe (¿ya fue eliminada?).")
    try:
        with conectar() as conexion:
            conexion.execute(f"DELETE FROM {NOMBRE_TABLA} WHERE id = ?", (id_consulta,))
    except sqlite3.Error as error:
        raise RuntimeError(f"No se pudo eliminar la consulta #{id_consulta}: {error}")


# --- Analítica del historial -------------------------------------------------

def obtener_resumen_estadistico() -> dict[str, Any]:
    """Métricas agregadas del historial para el panel de reportes.

    Devuelve un dict con:
      - total: número de registros.
      - por_tipo: dict {tipo_consulta: cantidad}.
      - tipo_mas_frecuente: str o None.
      - promedios_contaminantes: dict {contaminante: promedio}.
      - probabilidad_media: float o None (prob. media de alta contaminación).
      - pct_alta: float o None (% de consultas clasificadas como alta).
      - consultas_por_dia: DataFrame [fecha, cantidad] ordenado cronológicamente.
    """
    df = obtener_consultas()

    vacio: dict[str, Any] = {
        "total": 0,
        "por_tipo": {},
        "tipo_mas_frecuente": None,
        "promedios_contaminantes": {},
        "probabilidad_media": None,
        "pct_alta": None,
        "consultas_por_dia": pd.DataFrame(columns=["fecha", "cantidad"]),
    }
    if df.empty:
        return vacio

    por_tipo = (
        df["tipo_consulta"].fillna("Sin tipo").value_counts().to_dict()
        if "tipo_consulta" in df else {}
    )

    # `pd.to_numeric(errors="coerce")` protege contra filas migradas desde un
    # esquema legado (ver `_consolidar_esquema_legado`) que pudieran traer un
    # valor no numérico arrastrado (p. ej. un texto de error de sensor): se
    # trata como NaN en vez de tumbar el panel de reportes con un TypeError.
    promedios = {
        contaminante: round(float(pd.to_numeric(df[contaminante], errors="coerce").dropna().mean()), 2)
        for contaminante in CONTAMINANTES
        if contaminante in df and pd.to_numeric(df[contaminante], errors="coerce").notna().any()
    }

    probabilidad_num = pd.to_numeric(df["probabilidad"], errors="coerce") if "probabilidad" in df else None
    probabilidad_media = (
        round(float(probabilidad_num.dropna().mean()), 4)
        if probabilidad_num is not None and probabilidad_num.notna().any() else None
    )
    pct_alta = (
        round(100.0 * float((df["clase"] == 1).mean()), 1)
        if "clase" in df and df["clase"].notna().any() else None
    )

    fechas = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
    consultas_por_dia = (
        fechas.dropna().value_counts().sort_index().rename_axis("fecha")
        .reset_index(name="cantidad")
    )

    return {
        "total": int(len(df)),
        "por_tipo": por_tipo,
        "tipo_mas_frecuente": max(por_tipo, key=por_tipo.get) if por_tipo else None,
        "promedios_contaminantes": promedios,
        "probabilidad_media": probabilidad_media,
        "pct_alta": pct_alta,
        "consultas_por_dia": consultas_por_dia,
    }


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