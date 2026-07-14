"""Modelado y XAI (Rol B) — clasificación de horas de alta contaminación por PM2.5.

Este módulo consume el pipeline de Rol A (`preprocessing.cargar_y_limpiar`) y
entrena/evalúa los modelos del Panel 2 del dashboard:

    etiqueta binaria  ->  RF vs XGBoost  ->  matriz de confusión + métricas  ->  SHAP

Decisiones de diseño (justificadas frente a la rúbrica):

* **Etiqueta** ``y = (pm_25 > ECA_PM25)``. `ECA_PM25` es la ÚNICA fuente del umbral,
  para poder cambiarlo en vivo (pregunta típica del profesor) sin tocar el resto.
* **Features** = los contaminantes SALVO `pm_25`. Se derivan programáticamente de
  `CONTAMINANTES` para que sea imposible filtrar la variable objetivo (fuga de datos).
  No se usa `estacion` como feature (coherente con el clustering de Rol A, que solo
  la usa para interpretar).
* **Solo etiqueta real** (`pm_25_imputado == False`): no se entrena ni evalúa sobre
  PM2.5 imputado por climatología; si no, se mediría "qué tan bien se imputó", no la
  capacidad predictiva (ver CONTEXTO_ROL_A, sección 4).
* **Sin escalado**: RF y XGBoost son invariantes a transformaciones monótonas de las
  features; a diferencia del clustering de Rol A (que sí usa StandardScaler), aquí
  escalar no aporta y complica el pipeline de inferencia.
* **Desbalance ~92/8**: se comparan `class_weight='balanced'` (RF) / `scale_pos_weight`
  (XGB) frente a **SMOTE**, reportando el efecto en el recall de la clase minoritaria.
* **Reproducibilidad**: `SEED` se importa de `preprocessing` (fuente única del proyecto).

Uso rápido (regenera todos los artefactos del Panel 2 y del Reporte PDF)::

    uv run python src/models.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Rol A es la fuente única de SEED, CONTAMINANTES y del pipeline de limpieza.
from preprocessing import cargar_y_limpiar, CONTAMINANTES, SEED

# --------------------------------------------------------------------------
# Rutas (robustas al directorio de trabajo: se anclan al repo, no al CWD)
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "data" / "air_contamination.csv"
DIR_MODELOS = BASE_DIR / "models"

# --------------------------------------------------------------------------
# Constantes oficiales de Rol B (candidatas del "mapa de parámetros" del equipo)
# --------------------------------------------------------------------------

OBJETIVO = "pm_25"
"""Contaminante del que se deriva la etiqueta. No entra como feature."""

ECA_PM25 = 50.0
"""Umbral (µg/m³) que define 'hora de alta contaminación': y = (pm_25 > ECA_PM25).

Valor fijado por la división de roles del trabajo. Nota: el ECA peruano vigente para
PM2.5 (media 24h, DS 003-2017-MINAM) suele citarse en 25 µg/m³; 50 es una banda más
exigente. Se deja como constante ÚNICA para poder justificarlo o cambiarlo en vivo.
"""

FEATURES = [c for c in CONTAMINANTES if c != OBJETIVO]
"""Features del modelo: los contaminantes salvo el objetivo -> ['pm_10','so2','no2','o3','co'].

Se derivan de CONTAMINANTES para que sea estructuralmente imposible incluir `pm_25`
(evita fuga de la variable objetivo)."""

TEST_SIZE = 0.30
"""Proporción de test (split 70/30). La rúbrica pide poder cambiarlo en vivo (p.ej. 90/10)."""

UMBRAL_DECISION = 0.50
"""Umbral de probabilidad para clasificar como positivo. Parametrizable: bajarlo (p.ej.
0.30) sube el recall de la clase minoritaria a costa de más falsos positivos."""

SOLO_REAL = True
"""Si True, se usa solo PM2.5 medido (`pm_25_imputado == False`) para etiqueta y evaluación."""

np.random.seed(SEED)


# --------------------------------------------------------------------------
# Construcción del dataset de modelado
# --------------------------------------------------------------------------


def construir_dataset_modelado(
    df: pd.DataFrame,
    umbral: float = ECA_PM25,
    solo_real: bool = SOLO_REAL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Deriva ``(X, y)`` a partir del DataFrame limpio de Rol A.

    Parameters
    ----------
    df:
        Salida de `preprocessing.cargar_y_limpiar` (debe tener las columnas
        de `CONTAMINANTES` y `{objetivo}_imputado`).
    umbral:
        Umbral en µg/m³ para la etiqueta binaria (ver `ECA_PM25`).
    solo_real:
        Si True, descarta filas donde `pm_25` fue imputado (ver `SOLO_REAL`).

    Returns
    -------
    (X, y):
        `X` con columnas `FEATURES` (sin `pm_25`); `y` binaria (1 = alta
        contaminación).
    """
    base = df
    if solo_real:
        col_imputado = f"{OBJETIVO}_imputado"
        base = df[~df[col_imputado]].copy()

    y = (base[OBJETIVO] > umbral).astype(int)
    X = base[FEATURES].copy()

    # Barreras anti-fuga: el objetivo y su bandera NO pueden estar entre features.
    assert OBJETIVO not in X.columns, "Fuga: el objetivo está entre las features."
    assert not any(
        c.startswith(OBJETIVO) for c in X.columns
    ), "Fuga: hay columnas derivadas del objetivo entre las features."

    y.name = "alta_contaminacion"
    return X, y


def dividir(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """`train_test_split` estratificado y reproducible (`random_state=SEED`)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )


# --------------------------------------------------------------------------
# Balanceo del desbalance de clases
# --------------------------------------------------------------------------


def aplicar_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Sobremuestreo sintético de la clase minoritaria con SMOTE (solo en TRAIN).

    SMOTE nunca se aplica al conjunto de test: eso inflaría artificialmente las
    métricas. `random_state=SEED` para reproducibilidad.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Falta 'imbalanced-learn'. Instálalo con `uv sync` "
            "(está declarado en pyproject.toml)."
        ) from e

    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    return X_res, y_res


# --------------------------------------------------------------------------
# Entrenamiento de modelos
# --------------------------------------------------------------------------


def entrenar_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    balanceo: str = "class_weight",
) -> Any:
    """Entrena un Random Forest.

    Parameters
    ----------
    balanceo:
        ``'class_weight'`` usa `class_weight='balanced'`; cualquier otro valor
        (p.ej. ``'ninguno'`` cuando ya se aplicó SMOTE afuera) entrena sin pesos.

    Notes
    -----
    RF es reproducible con `random_state` incluso con `n_jobs=-1` (cada árbol
    recibe su propia semilla derivada); el paralelismo no rompe el determinismo.
    """
    from sklearn.ensemble import RandomForestClassifier

    class_weight = "balanced" if balanceo == "class_weight" else None
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=SEED,
        n_jobs=-1,
        class_weight=class_weight,
    )
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    balanceo: str = "scale_pos_weight",
) -> Any:
    """Entrena un XGBoost.

    Parameters
    ----------
    balanceo:
        ``'scale_pos_weight'`` fija ``scale_pos_weight = n_neg / n_pos`` (equivalente
        a `class_weight` para XGBoost); cualquier otro valor lo deja en 1.0.

    Notes
    -----
    Para el desbalance se prefiere `scale_pos_weight` a SMOTE por defecto: es el
    mecanismo nativo del modelo y no genera filas sintéticas. `random_state=SEED`
    y `tree_method='hist'` con la misma máquina/lock dan resultados reproducibles.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Falta 'xgboost'. Instálalo con `uv sync` "
            "(está declarado en pyproject.toml)."
        ) from e

    if balanceo == "scale_pos_weight":
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
    else:
        scale_pos_weight = 1.0

    modelo = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    return modelo


# --------------------------------------------------------------------------
# Predicción y evaluación
# --------------------------------------------------------------------------


def predecir(
    modelo: Any, X: pd.DataFrame, umbral: float = UMBRAL_DECISION
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(y_pred, y_proba)`` aplicando el umbral de decisión.

    Separar la probabilidad del corte hace explícito el `umbral`: cambiarlo (p.ej.
    a 0.30) reetiqueta sin reentrenar, respondiendo a la pregunta de "ajusta el
    umbral" de la rúbrica.
    """
    y_proba = modelo.predict_proba(X)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)
    return y_pred, y_proba


def evaluar(
    modelo: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    umbral: float = UMBRAL_DECISION,
    nombre: str = "",
) -> dict[str, Any]:
    """Calcula matriz de confusión y métricas (por clase y globales).

    Returns
    -------
    dict con: ``nombre``, ``matriz_confusion`` (tn, fp, fn, tp), ``accuracy``,
    ``roc_auc``, ``precision``/``recall``/``f1`` por clase, y ``umbral``.
    """
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        accuracy_score,
    )

    y_pred, y_proba = predecir(modelo, X_test, umbral)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    return {
        "nombre": nombre,
        "umbral": umbral,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "clase_0_baja": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
        },
        "clase_1_alta": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
        },
    }


def comparar_modelos(resultados: dict[str, dict]) -> pd.DataFrame:
    """Tabla comparativa (una fila por modelo) para el Panel 2 y el Reporte PDF.

    El foco está en la clase 1 (alta contaminación), que es la relevante y la
    minoritaria: F1 y recall de esa clase, más ROC-AUC y accuracy globales.
    """
    filas = []
    for clave, r in resultados.items():
        filas.append(
            {
                "modelo": clave,
                "f1_alta": round(r["clase_1_alta"]["f1"], 4),
                "recall_alta": round(r["clase_1_alta"]["recall"], 4),
                "precision_alta": round(r["clase_1_alta"]["precision"], 4),
                "roc_auc": round(r["roc_auc"], 4),
                "accuracy": round(r["accuracy"], 4),
            }
        )
    tabla = pd.DataFrame(filas).set_index("modelo")
    return tabla.sort_values("f1_alta", ascending=False)


# --------------------------------------------------------------------------
# Figuras: matriz de confusión y SHAP
# --------------------------------------------------------------------------


def figura_matriz_confusion(
    resultado: dict, ruta_png: Path, titulo: str | None = None
) -> None:
    """Guarda la matriz de confusión anotada (TP/TN/FP/FN) como PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = resultado["matriz_confusion"]
    mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    etiquetas = np.array(
        [[f"TN\n{m['tn']}", f"FP\n{m['fp']}"], [f"FN\n{m['fn']}", f"TP\n{m['tp']}"]]
    )

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, etiquetas[i, j], ha="center", va="center",
                color="white" if mat[i, j] > mat.max() / 2 else "black",
                fontsize=12, fontweight="bold",
            )
    ax.set_xticks([0, 1], ["Pred. baja", "Pred. alta"])
    ax.set_yticks([0, 1], ["Real baja", "Real alta"])
    ax.set_title(titulo or resultado.get("nombre", "Matriz de confusión"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=130)
    plt.close(fig)


def explicar_shap(
    modelo: Any,
    X: pd.DataFrame,
    dir_salida: Path,
    prefijo: str,
    n_muestra: int = 500,
    idx_instancia: int = 0,
) -> dict[str, str]:
    """Genera `summary_plot` (global) y `force_plot` (local) con SHAP.

    Se pasa `numpy` (no DataFrame) a SHAP para evitar fricción con pandas 3.x.
    Devuelve las rutas de los PNG generados.
    """
    try:
        import shap
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Falta 'shap'. Instálalo con `uv sync` (está declarado en pyproject.toml)."
        ) from e
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Muestra reproducible para acotar el coste de SHAP en datasets grandes.
    if len(X) > n_muestra:
        idx = np.random.RandomState(SEED).choice(len(X), size=n_muestra, replace=False)
        X_s = X.iloc[idx]
    else:
        X_s = X
    X_np = X_s.to_numpy()

    print("1. Importando SHAP...")
    explainer = shap.TreeExplainer(modelo)
    print("2. Calculando SHAP values...")
    valores = explainer.shap_values(X_np)

    # Normaliza a los valores SHAP de la clase positiva (RF -> lista/3D; XGB -> 2D).
    if isinstance(valores, list):
        sv = valores[1]
    elif getattr(valores, "ndim", 2) == 3:
        sv = valores[:, :, 1]
    else:
        sv = valores
    base = explainer.expected_value
    base = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) else base

    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta_summary = dir_salida / f"{prefijo}_shap_summary.png"
    ruta_force = dir_salida / f"{prefijo}_shap_force.png"

    # Global: qué variables empujan la predicción hacia 'alta contaminación'.
    print("3. Generando summary_plot...")
    plt.figure()
    shap.summary_plot(sv, X_s, feature_names=FEATURES, show=False)
    plt.tight_layout()
    print("4. Guardando summary...")
    plt.savefig(ruta_summary, dpi=130, bbox_inches="tight")
    plt.close()

    # Local: explicación de una instancia concreta (para la pregunta "¿por qué X?").
    print("5. Generando force_plot...")
    plt.figure()
    shap.force_plot(
        base, sv[idx_instancia], X_s.iloc[idx_instancia].round(2),
        feature_names=FEATURES, matplotlib=True, show=False,
    )
    print("6. Guardando force...")
    plt.savefig(ruta_force, dpi=130, bbox_inches="tight")
    plt.close()
    print("7. SHAP terminado.")

    return {"summary": str(ruta_summary), "force": str(ruta_force)}


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------


def guardar_modelo(modelo: Any, ruta: Path) -> None:
    """Serializa un modelo con joblib (formato estable para sklearn/xgboost)."""
    import joblib

    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta: Path) -> Any:
    """Carga un modelo serializado con `guardar_modelo`."""
    import joblib

    return joblib.load(ruta)


# --------------------------------------------------------------------------
# Orquestación reproducible
# --------------------------------------------------------------------------


def entrenar_y_evaluar_todo(
    df: pd.DataFrame,
    umbral_eca: float = ECA_PM25,
    umbral_decision: float = UMBRAL_DECISION,
    test_size: float = TEST_SIZE,
) -> dict[str, Any]:
    """Entrena RF (class_weight), XGBoost (scale_pos_weight) y RF+SMOTE; evalúa los tres.

    Es el punto de entrada que usan tanto `__main__` como el Panel 2. Devuelve un
    dict con modelos, métricas, tabla comparativa, el nombre del mejor modelo y los
    conjuntos de test (para SHAP y para la matriz de confusión).
    """
    X, y = construir_dataset_modelado(df, umbral=umbral_eca)
    X_train, X_test, y_train, y_test = dividir(X, y, test_size=test_size)

    # 1) RF con class_weight='balanced'
    rf = entrenar_rf(X_train, y_train, balanceo="class_weight")
    # 2) XGBoost con scale_pos_weight
    xgb = entrenar_xgb(X_train, y_train, balanceo="scale_pos_weight")
    # 3) RF + SMOTE (para mostrar el efecto del sobremuestreo en el recall)
    X_smote, y_smote = aplicar_smote(X_train, y_train)
    rf_smote = entrenar_rf(X_smote, y_smote, balanceo="ninguno")

    modelos = {
        "rf_classweight": rf,
        "xgb_scaleposw": xgb,
        "rf_smote": rf_smote,
    }
    resultados = {
        clave: evaluar(m, X_test, y_test, umbral=umbral_decision, nombre=clave)
        for clave, m in modelos.items()
    }
    tabla = comparar_modelos(resultados)
    mejor = tabla.index[0]

    return {
        "modelos": modelos,
        "resultados": resultados,
        "tabla": tabla,
        "mejor": mejor,
        "X_test": X_test,
        "y_test": y_test,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "balance_positivos_pct": float(y.mean() * 100),
    }


def _construir_metrics_json(salida: dict[str, Any]) -> dict[str, Any]:
    """Arma el dict serializable que se persiste en models/metrics.json."""
    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "umbral_eca_pm25": ECA_PM25,
        "umbral_decision": UMBRAL_DECISION,
        "test_size": TEST_SIZE,
        "solo_real": SOLO_REAL,
        "features": FEATURES,
        "objetivo": OBJETIVO,
        "n_train": salida["n_train"],
        "n_test": salida["n_test"],
        "balance_positivos_pct": round(salida["balance_positivos_pct"], 3),
        "mejor_modelo": salida["mejor"],
        "modelos": salida["resultados"],
    }


if __name__ == "__main__":
    print("Cargando y limpiando datos (Rol A)...")
    df = cargar_y_limpiar(str(RUTA_DATOS))

    print("Entrenando y evaluando RF / XGBoost / RF+SMOTE...")
    salida = entrenar_y_evaluar_todo(df)

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    # Persistir modelos
    guardar_modelo(salida["modelos"]["rf_classweight"], DIR_MODELOS / "rf.pkl")
    guardar_modelo(salida["modelos"]["xgb_scaleposw"], DIR_MODELOS / "xgb.pkl")
    guardar_modelo(salida["modelos"]["rf_smote"], DIR_MODELOS / "rf_smote.pkl")

    # Métricas auditables (versionadas en git)
    metrics = _construir_metrics_json(salida)
    with open(DIR_MODELOS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Figura de matriz de confusión del mejor modelo
    mejor = salida["mejor"]
    figura_matriz_confusion(
        salida["resultados"][mejor],
        DIR_MODELOS / "confusion_matrix.png",
        titulo=f"Matriz de confusión — {mejor}",
    )

    # SHAP del mejor modelo
    print(f"Generando SHAP para el mejor modelo ({mejor})...")
    explicar_shap(
        salida["modelos"][mejor], salida["X_test"], DIR_MODELOS, prefijo=mejor
    )

    print("\n=== Comparación de modelos (ordenada por F1 de la clase 'alta') ===")
    print(salida["tabla"].to_string())
    print(f"\nMejor modelo: {mejor}")
    print(f"Artefactos escritos en: {DIR_MODELOS}")
