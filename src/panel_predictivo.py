"""Panel 2 del dashboard — Predictivo.

Expone `render(df)` con el contenido del panel y `predecir_desde_entrada(modelo,
valores)`, reutilizada por el CRUD del Panel 4 para predecir con el mismo modelo.

Estrategia de carga: primero intenta leer los artefactos ya entrenados de `models/`
(rápido, ideal para deploy); si no existen, los entrena una sola vez y los cachea.
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
# Helpers sin dependencia de Streamlit (reutilizables por app.py y el CRUD)
# --------------------------------------------------------------------------


def predecir_desde_entrada(
    modelo: Any, valores: dict[str, float], umbral: float = M.UMBRAL_DECISION
) -> dict[str, Any]:
    """Predice a partir de un dict con los 5 contaminantes de entrada.

    Recibe `{pm_10, so2, no2, o3, co}` y devuelve `{etiqueta, clase, probabilidad,
    umbral}` — lo que el CRUD del Panel 4 persiste como "entrada + predicción".

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

    @st.cache_resource(show_spinner=False)
    def _cache(_df: pd.DataFrame):
        artefactos = _cargar_artefactos()
        if artefactos is not None:
            return {"origen": "disco", **artefactos}
        # Fallback: entrenar en caliente (cold start). Para deploy conviene
        # commitear models/*.pkl y evitar este costo.
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
    import altair as alt
    import streamlit as st

    import theme

    if df is None:
        df = cargar_y_limpiar(str(M.RUTA_DATOS))

    st.header(":material/smart_toy: Panel 2 — Predictivo: alta contaminación por PM2.5")

    with st.skeleton(height=160):
        todo = _obtener_todo(df)
    metrics = todo["metrics"]
    mejor = metrics["mejor_modelo"]

    with st.container(border=True):
        st.subheader(":material/info: Planteamiento del problema")
        st.markdown(
            f"""
Clasificar cada hora como *alta contaminación* cuando
`PM2.5 > {metrics['umbral_eca_pm25']:.0f} µg/m³`, usando los otros contaminantes
({', '.join(metrics['features'])}) como variables. Se excluye `pm_25` para evitar
fuga de la variable objetivo, y se entrena solo con PM2.5 **medido** (no imputado).
La clase 'alta' es minoritaria (~{metrics['balance_positivos_pct']:.1f}%), de ahí el
manejo explícito del desbalance.
            """
        )

    # --- Comparación de modelos -------------------------------------------------
    with st.container(border=True):
        st.subheader(":material/bar_chart: Comparación de modelos")
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
        st.dataframe(tabla, width="stretch")
        st.caption(
            f"Mejor modelo por F1 de la clase minoritaria: **{mejor}**. "
            "Se justifica priorizar F1/recall de 'alta' porque el costo de no avisar una "
            "hora contaminada (FN) es mayor que una falsa alarma (FP)."
        )

    # --- Matriz de confusión (reactiva) + SHAP (imágenes generadas por models.py) ---
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(":material/grid_view: Matriz de confusión")
            p = theme.paleta()
            cm = metrics["modelos"][mejor]["matriz_confusion"]
            orden = ["Baja (0)", "Alta (1)"]
            datos_cm = pd.DataFrame([
                {"real": "Baja (0)", "predicha": "Baja (0)", "conteo": cm["tn"]},
                {"real": "Baja (0)", "predicha": "Alta (1)", "conteo": cm["fp"]},
                {"real": "Alta (1)", "predicha": "Baja (0)", "conteo": cm["fn"]},
                {"real": "Alta (1)", "predicha": "Alta (1)", "conteo": cm["tp"]},
            ])
            fondo_cm = alt.Chart(datos_cm).mark_rect().encode(
                x=alt.X("predicha:N", title="Clase predicha", sort=orden),
                y=alt.Y("real:N", title="Clase real", sort=orden),
                color=alt.Color("conteo:Q", title="Conteo",
                                 scale=alt.Scale(range=[p["SUPERFICIE"], p["MAUVE"]])),
                tooltip=[alt.Tooltip("real:N", title="Clase real"),
                         alt.Tooltip("predicha:N", title="Clase predicha"),
                         alt.Tooltip("conteo:Q", title="Conteo")],
            )
            texto_cm = alt.Chart(datos_cm).mark_text(fontSize=16, fontWeight="bold").encode(
                x=alt.X("predicha:N", sort=orden), y=alt.Y("real:N", sort=orden),
                text="conteo:Q", color=alt.value(p["TEXTO"]),
            )
            chart_cm = (fondo_cm + texto_cm).properties(height=280)
            st.altair_chart(theme.aplicar_estilo_altair(chart_cm), theme=None, width="stretch")
            st.caption(
                "Filas = clase real, columnas = clase predicha. La diagonal son los "
                "aciertos; fuera de la diagonal, los errores (falsos positivos y falsos "
                "negativos) del modelo ganador."
            )
        with col2:
            st.subheader(":material/science: Importancia global (SHAP)")
            summ = M.DIR_MODELOS / f"{mejor}_shap_summary.png"
            if summ.exists():
                st.image(str(summ), width="stretch")
                st.caption(
                    "Cada punto es una hora del set de test. Arriba, los contaminantes que "
                    "más mueven la predicción en general; el color indica si ese contaminante "
                    "estaba alto (rojo) o bajo (azul) en esa hora."
                )
            else:
                st.info("SHAP summary se genera con `uv run python src/models.py`.")

    force = M.DIR_MODELOS / f"{mejor}_shap_force.png"
    if force.exists():
        with st.container(border=True):
            st.subheader(":material/track_changes: Explicación local (SHAP force plot)")
            st.image(str(force), width="stretch")
            st.caption(
                "Cómo se arma la predicción para UN caso puntual: cada contaminante empuja "
                "la probabilidad hacia 'alta contaminación' (rojo) o hacia 'baja' (azul) "
                "desde un valor base, hasta llegar a la probabilidad final del modelo."
            )

    # --- Predicción interactiva (alimenta el CRUD del Panel 4) ------------------
    with st.container(border=True):
        st.subheader(":material/play_circle: Probar una predicción")
        umbral = st.slider(
            "Umbral de decisión", 0.05, 0.95, float(metrics["umbral_decision"]), 0.05,
            help="Bajarlo sube el recall de 'alta' a costa de más falsos positivos.",
        )
        modelo = todo["rf"] if mejor.startswith("rf") else todo["xgb"]

        st.caption(
            "Ingresa una lectura de los otros contaminantes para esa hora y el modelo "
            "estima si esa combinación corresponde a una hora de alta contaminación de PM2.5."
        )
        defaults = {"pm_10": 80.0, "so2": 10.0, "no2": 30.0, "o3": 15.0, "co": 800.0}
        with st.form("form_prediccion_interactiva", border=False):
            cols = st.columns(len(M.FEATURES))
            valores: dict[str, float] = {}
            for c, feat in zip(cols, M.FEATURES):
                valores[feat] = c.number_input(feat, min_value=0.0, value=defaults.get(feat, 0.0))
            enviado = st.form_submit_button("Predecir", type="primary")

        if enviado:
            res = predecir_desde_entrada(modelo, valores, umbral=umbral)
            etiqueta = res["etiqueta"]
            (st.error if res["clase"] == 1 else st.success)(
                f"{etiqueta} · probabilidad={res['probabilidad']:.3f} (umbral={umbral:.2f})"
            )
            # Guardado en session_state para que el CRUD del Panel 4 lo reutilice.
            st.session_state["ultima_prediccion"] = {**valores, **res}

    if todo["origen"] == "entrenado":
        st.caption(
            ":material/warning: Modelos entrenados en caliente (no había artefactos en `models/`). "
            "Para el deploy, commitea `models/*.pkl` o corre `models.py` antes."
        )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Panel 2 — Predictivo", layout="wide")
    render()


if __name__ == "__main__":
    main()
