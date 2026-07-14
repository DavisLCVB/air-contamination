"""Panel 2 del dashboard — Predictivo (Rol B).

Entrega el CONTENIDO del Panel 2 como una función `render(df)` para que Rol C la
integre en `app.py` sin acoplarse a los detalles del modelado::

    # dentro de app.py (Rol C)
    from panel_predictivo import render
    with tab_predictivo:
        render(df)          # df = salida de preprocessing.cargar_y_limpiar

También expone `predecir_desde_entrada(modelo, valores)` para que Rol D (Panel 4 CRUD)
guarde "entrada + predicción devuelta" reutilizando exactamente el mismo modelo.

Ejecución aislada (demo del Panel 2 sin la app integrada)::

    uv run streamlit run src/panel_predictivo.py

Estrategia de carga: primero intenta leer los artefactos ya entrenados de `models/`
(rápido, ideal para deploy). Si no existen, los entrena una sola vez y los cachea.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import models as M
from preprocessing import cargar_y_limpiar

RUTA_METRICS = M.DIR_MODELOS / "metrics.json"
RUTA_RF = M.DIR_MODELOS / "rf.pkl"
RUTA_XGB = M.DIR_MODELOS / "xgb.pkl"


# --------------------------------------------------------------------------
# Helpers SIN dependencia de Streamlit (reutilizables por Rol C y Rol D)
# --------------------------------------------------------------------------


def predecir_desde_entrada(
    modelo: Any, valores: dict[str, float], umbral: float = M.UMBRAL_DECISION
) -> dict[str, Any]:
    """Predice a partir de un dict con los 5 contaminantes de entrada.

    Contrato pensado para el CRUD de Rol D: recibe `{pm_10, so2, no2, o3, co}`
    y devuelve `{etiqueta, clase, probabilidad, umbral}` — lo que se persiste como
    "entrada + predicción devuelta".

    Parameters
    ----------
    modelo:
        Modelo entrenado (RF o XGBoost) cargado desde `models/`.
    valores:
        Dict con una clave por cada nombre en `models.FEATURES`.
    umbral:
        Umbral de decisión (ver `models.UMBRAL_DECISION`).
    """
    faltan = [f for f in M.FEATURES if f not in valores]
    if faltan:
        raise ValueError(f"Faltan features en la entrada: {faltan}")

    X = pd.DataFrame([[valores[f] for f in M.FEATURES]], columns=M.FEATURES)
    y_pred, y_proba = M.predecir(modelo, X, umbral=umbral)
    clase = int(y_pred[0])
    return {
        "clase": clase,
        "etiqueta": "Alta contaminación" if clase == 1 else "Baja contaminación",
        "probabilidad": float(y_proba[0]),
        "umbral": umbral,
    }


def _cargar_artefactos() -> dict[str, Any] | None:
    """Devuelve {rf, xgb, metrics} si los artefactos existen en disco; si no, None."""
    if RUTA_RF.exists() and RUTA_XGB.exists() and RUTA_METRICS.exists():
        return {
            "rf": M.cargar_modelo(RUTA_RF),
            "xgb": M.cargar_modelo(RUTA_XGB),
            "metrics": json.loads(RUTA_METRICS.read_text(encoding="utf-8")),
        }
    return None


# --------------------------------------------------------------------------
# UI de Streamlit
# --------------------------------------------------------------------------


def _obtener_todo(df: pd.DataFrame):
    """Carga artefactos de `models/` o, como fallback, entrena una sola vez (cacheado)."""
    import streamlit as st

    @st.cache_resource(show_spinner="Preparando modelos del Panel 2...")
    def _cache(_df: pd.DataFrame):
        artefactos = _cargar_artefactos()
        if artefactos is not None:
            return {"origen": "disco", **artefactos}
        # Fallback: entrenar en caliente (cold start). Recomendado para deploy:
        # commitear models/*.pkl para evitar este costo (ver CONTEXTO_ROL_B.md).
        salida = M.entrenar_y_evaluar_todo(_df)
        return {
            "origen": "entrenado",
            "rf": salida["modelos"]["rf_classweight"],
            "xgb": salida["modelos"]["xgb_scaleposw"],
            "metrics": M._construir_metrics_json(salida),
        }

    return _cache(df)


def render(df: pd.DataFrame | None = None) -> None:
    """Renderiza el Panel 2 (predictivo) dentro de la app de Streamlit."""
    import streamlit as st

    if df is None:
        df = cargar_y_limpiar(str(M.RUTA_DATOS))

    todo = _obtener_todo(df)
    metrics = todo["metrics"]
    mejor = metrics["mejor_modelo"]

    st.header("Panel 2 — Predictivo: alta contaminación por PM2.5")
    st.markdown(
        f"""
**Problema.** Clasificar cada hora como *alta contaminación* cuando
`PM2.5 > {metrics['umbral_eca_pm25']:.0f} µg/m³`, usando los otros contaminantes
({', '.join(metrics['features'])}) como variables. Se excluye `pm_25` para evitar
fuga de la variable objetivo, y se entrena solo con PM2.5 **medido** (no imputado).
La clase 'alta' es minoritaria (~{metrics['balance_positivos_pct']:.1f}%), de ahí el
manejo explícito del desbalance.
        """
    )

    # --- Comparación de modelos -------------------------------------------------
    st.subheader("Comparación de modelos")
    filas = []
    for clave, r in metrics["modelos"].items():
        filas.append(
            {
                "modelo": clave,
                "F1 (alta)": round(r["clase_1_alta"]["f1"], 4),
                "Recall (alta)": round(r["clase_1_alta"]["recall"], 4),
                "Precision (alta)": round(r["clase_1_alta"]["precision"], 4),
                "ROC-AUC": round(r["roc_auc"], 4),
                "Accuracy": round(r["accuracy"], 4),
            }
        )
    tabla = pd.DataFrame(filas).set_index("modelo")
    st.dataframe(tabla, use_container_width=True)
    st.caption(
        f"Mejor modelo por F1 de la clase minoritaria: **{mejor}**. "
        "Se justifica priorizar F1/recall de 'alta' porque el costo de no avisar una "
        "hora contaminada (FN) es mayor que una falsa alarma (FP)."
    )

    # --- Matriz de confusión + SHAP (imágenes generadas por models.py) ----------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de confusión")
        cm = M.DIR_MODELOS / "confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), use_container_width=True)
        else:
            r = metrics["modelos"][mejor]["matriz_confusion"]
            st.write(r)
            st.info("Ejecuta `uv run python src/models.py` para generar la figura.")
    with col2:
        st.subheader("Importancia global (SHAP)")
        summ = M.DIR_MODELOS / f"{mejor}_shap_summary.png"
        if summ.exists():
            st.image(str(summ), use_container_width=True)
        else:
            st.info("SHAP summary se genera con `uv run python src/models.py`.")

    force = M.DIR_MODELOS / f"{mejor}_shap_force.png"
    if force.exists():
        st.subheader("Explicación local (SHAP force plot)")
        st.image(str(force), use_container_width=True)

    # --- Predicción interactiva (alimenta el CRUD de Rol D) ---------------------
    st.subheader("Probar una predicción")
    umbral = st.slider(
        "Umbral de decisión", 0.05, 0.95, float(metrics["umbral_decision"]), 0.05,
        help="Bajarlo sube el recall de 'alta' a costa de más falsos positivos.",
    )
    modelo = todo["rf"] if mejor.startswith("rf") else todo["xgb"]

    cols = st.columns(len(M.FEATURES))
    valores: dict[str, float] = {}
    defaults = {"pm_10": 80.0, "so2": 10.0, "no2": 30.0, "o3": 15.0, "co": 800.0}
    for c, feat in zip(cols, M.FEATURES):
        valores[feat] = c.number_input(feat, min_value=0.0, value=defaults.get(feat, 0.0))

    if st.button("Predecir"):
        res = predecir_desde_entrada(modelo, valores, umbral=umbral)
        etiqueta = res["etiqueta"]
        (st.error if res["clase"] == 1 else st.success)(
            f"{etiqueta} · probabilidad={res['probabilidad']:.3f} (umbral={umbral:.2f})"
        )
        # Se devuelve el dict para que Rol D lo persista en el CRUD (Panel 4).
        st.session_state["ultima_prediccion"] = {**valores, **res}

    if todo["origen"] == "entrenado":
        st.caption(
            "⚠ Modelos entrenados en caliente (no había artefactos en `models/`). "
            "Para el deploy, commitea `models/*.pkl` o corre `models.py` antes."
        )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Panel 2 — Predictivo", layout="wide")
    render()


if __name__ == "__main__":
    main()
