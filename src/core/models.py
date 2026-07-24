"""Clasificación de horas de alta contaminación por PM2.5 (RF / XGBoost + SHAP)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.preprocessing import cargar_y_limpiar, CONTAMINANTES, SEED

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RUTA_DATOS = BASE_DIR / "data" / "air_contamination.csv"
DIR_MODELOS = BASE_DIR / "models"
RUTA_METRICS = DIR_MODELOS / "metrics.json"
RUTA_RF = DIR_MODELOS / "rf.pkl"
RUTA_XGB = DIR_MODELOS / "xgb.pkl"

# --- Constantes (mapa de parámetros: ver docs/mapa_parametros.md) ------------

OBJETIVO = "pm_25"
ECA_PM25 = 50.0  # µg/m³ -- umbral de "hora de alta contaminación"
FEATURES = [c for c in CONTAMINANTES if c != OBJETIVO]
TEST_SIZE = 0.30
UMBRAL_DECISION = 0.50
SOLO_REAL = True  # solo entrena/evalúa sobre pm_25 medido, no imputado

np.random.seed(SEED)


# --- Dataset de modelado ------------------------------------------------------

def construir_dataset_modelado(
    df: pd.DataFrame, umbral: float = ECA_PM25, solo_real: bool = SOLO_REAL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Deriva (X, y) del DataFrame limpio; y = (pm_25 > umbral)."""
    base = df
    if solo_real:
        col_imputado = f"{OBJETIVO}_imputado"
        base = df[~df[col_imputado]].copy()

    y = (base[OBJETIVO] > umbral).astype(int)
    X = base[FEATURES].copy()

    assert OBJETIVO not in X.columns, "Fuga: el objetivo está entre las features."
    assert not any(c.startswith(OBJETIVO) for c in X.columns), "Fuga: columna derivada del objetivo."

    y.name = "alta_contaminacion"
    return X, y


def dividir(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, grupos: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split train/test; con `grupos` usa GroupShuffleSplit (evita fuga por autocorrelación horaria)."""
    if grupos is not None:
        from sklearn.model_selection import GroupShuffleSplit

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        idx_train, idx_test = next(gss.split(X, y, groups=grupos.loc[X.index]))
        return X.iloc[idx_train], X.iloc[idx_test], y.iloc[idx_train], y.iloc[idx_test]

    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


# --- Balanceo -----------------------------------------------------------------

def aplicar_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Sobremuestreo SMOTE de la clase minoritaria (solo en train)."""
    from imblearn.over_sampling import SMOTE

    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    return X_res, y_res


# --- Entrenamiento --------------------------------------------------------------

def entrenar_rf(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 300,
    balanceo: str = "class_weight", max_depth: int = 20, min_samples_leaf: int = 20,
) -> Any:
    """Random Forest; `balanceo='class_weight'` usa class_weight='balanced'."""
    from sklearn.ensemble import RandomForestClassifier

    class_weight = "balanced" if balanceo == "class_weight" else None
    modelo = RandomForestClassifier(
        n_estimators=n_estimators, random_state=SEED, n_jobs=-1,
        class_weight=class_weight, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
    )
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_xgb(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 300,
    learning_rate: float = 0.1, max_depth: int = 6, balanceo: str = "scale_pos_weight",
) -> Any:
    """XGBoost; `balanceo='scale_pos_weight'` fija scale_pos_weight = n_neg/n_pos."""
    from xgboost import XGBClassifier

    if balanceo == "scale_pos_weight":
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
    else:
        scale_pos_weight = 1.0

    modelo = XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=scale_pos_weight,
        tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    return modelo


# --- Predicción y evaluación -----------------------------------------------------

def predecir(modelo: Any, X: pd.DataFrame, umbral: float = UMBRAL_DECISION) -> tuple[np.ndarray, np.ndarray]:
    """(y_pred, y_proba) aplicando `umbral` sobre predict_proba."""
    y_proba = modelo.predict_proba(X)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)
    return y_pred, y_proba


def evaluar(
    modelo: Any, X_test: pd.DataFrame, y_test: pd.Series,
    umbral: float = UMBRAL_DECISION, nombre: str = "",
) -> dict[str, Any]:
    """Matriz de confusión + métricas por clase y globales."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, accuracy_score

    y_pred, y_proba = predecir(modelo, X_test, umbral)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)

    return {
        "nombre": nombre,
        "umbral": umbral,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "clase_0_baja": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0])},
        "clase_1_alta": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1])},
    }


def comparar_modelos(resultados: dict[str, dict]) -> pd.DataFrame:
    """Tabla comparativa por modelo, ordenada por F1 de la clase alta."""
    filas = []
    for clave, r in resultados.items():
        filas.append({
            "modelo": clave,
            "f1_alta": round(r["clase_1_alta"]["f1"], 4),
            "recall_alta": round(r["clase_1_alta"]["recall"], 4),
            "precision_alta": round(r["clase_1_alta"]["precision"], 4),
            "roc_auc": round(r["roc_auc"], 4),
            "accuracy": round(r["accuracy"], 4),
        })
    tabla = pd.DataFrame(filas).set_index("modelo")
    return tabla.sort_values("f1_alta", ascending=False)


# --- Persistencia ------------------------------------------------------------

def guardar_modelo(modelo: Any, ruta: Path) -> None:
    import joblib

    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta: Path) -> Any:
    import joblib

    return joblib.load(ruta)


def cargar_artefactos() -> dict[str, Any] | None:
    """{rf, xgb, metrics} desde disco si existen; None si falta alguno."""
    if RUTA_RF.exists() and RUTA_XGB.exists() and RUTA_METRICS.exists():
        return {
            "rf": cargar_modelo(RUTA_RF),
            "xgb": cargar_modelo(RUTA_XGB),
            "metrics": json.loads(RUTA_METRICS.read_text(encoding="utf-8")),
        }
    return None


def predecir_desde_entrada(
    modelo: Any, valores: dict[str, float], umbral: float = UMBRAL_DECISION
) -> dict[str, Any]:
    """Predice a partir de un dict {feature: valor} (contrato del CRUD y Panel 2)."""
    faltan = [f for f in FEATURES if f not in valores]
    if faltan:
        raise ValueError(f"Faltan features en la entrada: {faltan}")

    X = pd.DataFrame([[valores[f] for f in FEATURES]], columns=FEATURES)
    y_pred, y_proba = predecir(modelo, X, umbral=umbral)
    clase = int(y_pred[0])
    return {
        "clase": clase,
        "etiqueta": "Alta contaminación" if clase == 1 else "Baja contaminación",
        "probabilidad": float(y_proba[0]),
        "umbral": umbral,
    }


# --- Orquestación --------------------------------------------------------------

def entrenar_y_evaluar_todo(
    df: pd.DataFrame, umbral_eca: float = ECA_PM25,
    umbral_decision: float = UMBRAL_DECISION, test_size: float = TEST_SIZE,
) -> dict[str, Any]:
    """Entrena RF (class_weight), XGBoost (scale_pos_weight) y RF+SMOTE; evalúa los tres."""
    X, y = construir_dataset_modelado(df, umbral=umbral_eca)
    grupos = df.loc[X.index, "fecha_hora"].dt.normalize()  # split por día, no por fila
    X_train, X_test, y_train, y_test = dividir(X, y, test_size=test_size, grupos=grupos)

    rf = entrenar_rf(X_train, y_train, n_estimators=150, balanceo="class_weight")
    xgb = entrenar_xgb(X_train, y_train, balanceo="scale_pos_weight")
    X_smote, y_smote = aplicar_smote(X_train, y_train)
    rf_smote = entrenar_rf(X_smote, y_smote, n_estimators=150, balanceo="ninguno")

    modelos = {"rf_classweight": rf, "xgb_scaleposw": xgb, "rf_smote": rf_smote}
    resultados = {
        clave: evaluar(m, X_test, y_test, umbral=umbral_decision, nombre=clave)
        for clave, m in modelos.items()
    }
    tabla = comparar_modelos(resultados)
    # rf_smote no se embarca en el deploy (ver .gitignore); "mejor" nunca debe apuntar a él
    candidatos_desplegables = tabla.loc[tabla.index != "rf_smote"]
    mejor = candidatos_desplegables.index[0]

    return {
        "modelos": modelos, "resultados": resultados, "tabla": tabla, "mejor": mejor,
        "X_test": X_test, "y_test": y_test,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "balance_positivos_pct": float(y.mean() * 100),
    }


def resolver_modelos(df: pd.DataFrame) -> dict[str, Any]:
    """Carga artefactos de `models/` o, si faltan, entrena en caliente (cold start)."""
    artefactos = cargar_artefactos()
    if artefactos is not None:
        return {"origen": "disco", **artefactos}
    salida = entrenar_y_evaluar_todo(df)
    return {
        "origen": "entrenado",
        "rf": salida["modelos"]["rf_classweight"],
        "xgb": salida["modelos"]["xgb_scaleposw"],
        "metrics": construir_metrics_json(salida),
    }


def construir_metrics_json(salida: dict[str, Any]) -> dict[str, Any]:
    """Dict serializable persistido en models/metrics.json."""
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
    from application.graficos import figura_matriz_confusion, explicar_shap

    print("Cargando y limpiando datos...")
    df = cargar_y_limpiar(str(RUTA_DATOS))

    print("Entrenando y evaluando RF / XGBoost / RF+SMOTE...")
    salida = entrenar_y_evaluar_todo(df)

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    guardar_modelo(salida["modelos"]["rf_classweight"], DIR_MODELOS / "rf.pkl")
    guardar_modelo(salida["modelos"]["xgb_scaleposw"], DIR_MODELOS / "xgb.pkl")
    guardar_modelo(salida["modelos"]["rf_smote"], DIR_MODELOS / "rf_smote.pkl")

    metrics = construir_metrics_json(salida)
    with open(DIR_MODELOS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    mejor = salida["mejor"]
    figura_matriz_confusion(
        salida["resultados"][mejor], DIR_MODELOS / "confusion_matrix.png",
        titulo=f"Matriz de confusión — {mejor}",
    )

    print(f"Generando SHAP para el mejor modelo ({mejor})...")
    explicar_shap(salida["modelos"][mejor], salida["X_test"], DIR_MODELOS, prefijo=mejor)

    print("\n=== Comparación de modelos (ordenada por F1 de la clase 'alta') ===")
    print(salida["tabla"].to_string())
    print(f"\nMejor modelo: {mejor}")
    print(f"Artefactos escritos en: {DIR_MODELOS}")
