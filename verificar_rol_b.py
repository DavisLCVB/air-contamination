"""Verificación (smoke test) del Rol B — Modelado y XAI.

Ejecuta comprobaciones rápidas sobre una MUESTRA de los datos (segundos, no minutos)
y reporta PASS / FAIL / WARN por cada punto. No reemplaza a `models.py` (que entrena
con todo); sirve para responder "¿está todo bien instalado y conectado?".

Uso:
    uv run python verificar_rol_b.py

Código de salida 0 si no hay FAIL; 1 si hay al menos un FAIL.
"""

from __future__ import annotations

import sys
import traceback

sys.path.append("src")

import warnings

warnings.filterwarnings("ignore")

N_MUESTRA = 40_000          # filas para el smoke test (rápido)
N_ARBOLES_TEST = 60         # árboles reducidos para que sea veloz

_fallos = 0
_avisos = 0


def check(nombre: str, cond: bool, detalle: str = "", warn: bool = False) -> bool:
    global _fallos, _avisos
    if cond:
        print(f"  [PASS] {nombre}" + (f" — {detalle}" if detalle else ""))
    elif warn:
        _avisos += 1
        print(f"  [WARN] {nombre}" + (f" — {detalle}" if detalle else ""))
    else:
        _fallos += 1
        print(f"  [FAIL] {nombre}" + (f" — {detalle}" if detalle else ""))
    return cond


def seccion(titulo: str) -> None:
    print(f"\n=== {titulo} ===")


# --------------------------------------------------------------------------
seccion("0. Entorno / dependencias")
deps = {}
for paquete, modulo in [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"), ("shap", "shap"), ("numba", "numba"),
    ("imbalanced-learn", "imblearn"), ("joblib", "joblib"),
    ("matplotlib", "matplotlib"), ("streamlit", "streamlit"),
]:
    try:
        m = __import__(modulo)
        v = getattr(m, "__version__", "?")
        deps[modulo] = True
        check(f"import {paquete}", True, f"v{v}")
    except Exception as e:  # noqa: BLE001
        deps[modulo] = False
        check(f"import {paquete}", False, f"NO instalado ({e}). Corre `uv sync`.")

# --------------------------------------------------------------------------
seccion("1. Datos y pipeline de Rol A")
df = None
try:
    import models as M
    from preprocessing import cargar_y_limpiar

    df = cargar_y_limpiar(str(M.RUTA_DATOS))
    check("cargar_y_limpiar ejecuta", True, f"shape={df.shape}")
    nan_cont = int(df[M.CONTAMINANTES].isna().sum().sum())
    check("0% NaN en contaminantes", nan_cont == 0, f"NaN={nan_cont}")
    check("columna pm_25_imputado presente", "pm_25_imputado" in df.columns)
except Exception:  # noqa: BLE001
    check("cargar_y_limpiar ejecuta", False, "ver traceback abajo")
    traceback.print_exc()

# --------------------------------------------------------------------------
seccion("2. Dataset de modelado (etiqueta y anti-fuga)")
X = y = None
if df is not None:
    try:
        X, y = M.construir_dataset_modelado(df)
        check("features correctas", list(X.columns) == M.FEATURES, f"{list(X.columns)}")
        check("pm_25 NO está entre features", "pm_25" not in X.columns)
        check("sin columnas derivadas de pm_25",
              not any(c.startswith("pm_25") for c in X.columns))
        check("X sin NaN", int(X.isna().sum().sum()) == 0)
        bal = float(y.mean() * 100)
        check("balance de clase plausible (4–12% positivos)", 4 <= bal <= 12,
              f"{bal:.1f}% positivos", warn=not (4 <= bal <= 12))
        check("hay dos clases", set(y.unique()) == {0, 1})
    except Exception:  # noqa: BLE001
        check("construir_dataset_modelado", False, "ver traceback")
        traceback.print_exc()

# --------------------------------------------------------------------------
seccion("3. Muestra + split estratificado")
Xtr = Xte = ytr = yte = None
if X is not None:
    try:
        Xs = X.sample(n=min(N_MUESTRA, len(X)), random_state=M.SEED)
        ys = y.loc[Xs.index]
        Xtr, Xte, ytr, yte = M.dividir(Xs, ys)
        b_tr, b_te = ytr.mean() * 100, yte.mean() * 100
        check("split genera train y test", len(Xtr) > 0 and len(Xte) > 0,
              f"train={len(Xtr)} test={len(Xte)}")
        check("estratificación coherente (balance train≈test)",
              abs(b_tr - b_te) < 1.5, f"train={b_tr:.1f}% test={b_te:.1f}%")
    except Exception:  # noqa: BLE001
        check("split", False, "ver traceback")
        traceback.print_exc()

# --------------------------------------------------------------------------
seccion("4. Entrenamiento y evaluación (muestra, árboles reducidos)")
rf = None
if Xtr is not None:
    # Random Forest
    try:
        rf = M.entrenar_rf(Xtr, ytr, n_estimators=N_ARBOLES_TEST, balanceo="class_weight")
        r = M.evaluar(rf, Xte, yte)
        auc = r["roc_auc"]
        check("RF entrena y evalúa", True,
              f"ROC-AUC={auc:.3f} recall_alta={r['clase_1_alta']['recall']:.3f}")
        check("ROC-AUC en rango sano (0.70–0.999)", 0.70 <= auc <= 0.999,
              f"AUC={auc:.3f}", warn=True)
        check("NO parece fuga trivial (AUC<0.999 y acc<0.999)",
              auc < 0.999 and r["accuracy"] < 0.999,
              f"AUC={auc:.3f} acc={r['accuracy']:.3f}", warn=True)
        yp05 = M.predecir(rf, Xte, 0.50)[0]
        yp03 = M.predecir(rf, Xte, 0.30)[0]
        rec05 = ((yp05 == 1) & (yte.values == 1)).sum() / max((yte.values == 1).sum(), 1)
        rec03 = ((yp03 == 1) & (yte.values == 1)).sum() / max((yte.values == 1).sum(), 1)
        check("bajar umbral 0.50→0.30 sube (o iguala) recall", rec03 >= rec05,
              f"recall {rec05:.3f} → {rec03:.3f}")
    except Exception:  # noqa: BLE001
        check("RF entrena y evalúa", False, "ver traceback")
        traceback.print_exc()

    # XGBoost
    if deps.get("xgboost"):
        try:
            xgb = M.entrenar_xgb(Xtr, ytr, n_estimators=N_ARBOLES_TEST)
            rx = M.evaluar(xgb, Xte, yte)
            check("XGBoost entrena y evalúa", True, f"ROC-AUC={rx['roc_auc']:.3f}")
        except Exception:  # noqa: BLE001
            check("XGBoost entrena y evalúa", False, "ver traceback")
            traceback.print_exc()
    else:
        check("XGBoost entrena y evalúa", False, "xgboost no instalado")

    # SMOTE
    if deps.get("imblearn"):
        try:
            Xr, yr = M.aplicar_smote(Xtr, ytr)
            check("SMOTE balancea el train", yr.mean() > ytr.mean(),
                  f"positivos {ytr.mean()*100:.1f}% → {yr.mean()*100:.1f}%")
        except Exception:  # noqa: BLE001
            check("SMOTE balancea el train", False, "ver traceback")
            traceback.print_exc()
    else:
        check("SMOTE balancea el train", False, "imbalanced-learn no instalado")

# --------------------------------------------------------------------------
seccion("5. SHAP (muestra pequeña)")
if rf is not None and deps.get("shap"):
    try:
        import tempfile
        from pathlib import Path

        rutas = M.explicar_shap(rf, Xte, Path(tempfile.mkdtemp()), prefijo="smoke",
                                n_muestra=300)
        ok = Path(rutas["summary"]).exists() and Path(rutas["force"]).exists()
        check("SHAP genera summary + force plot", ok)
    except Exception:  # noqa: BLE001
        check("SHAP genera summary + force plot", False,
              "ver traceback (revisa versión de shap vs pandas)")
        traceback.print_exc()
elif not deps.get("shap"):
    check("SHAP genera summary + force plot", False, "shap no instalado")

# --------------------------------------------------------------------------
seccion("6. Persistencia (joblib)")
if rf is not None and deps.get("joblib"):
    try:
        import tempfile
        from pathlib import Path
        import numpy as np

        ruta = Path(tempfile.mkdtemp()) / "rf.pkl"
        M.guardar_modelo(rf, ruta)
        rf2 = M.cargar_modelo(ruta)
        igual = np.array_equal(rf.predict(Xte[:200]), rf2.predict(Xte[:200]))
        check("guardar/cargar modelo es idéntico", igual)
    except Exception:  # noqa: BLE001
        check("guardar/cargar modelo", False, "ver traceback")
        traceback.print_exc()

# --------------------------------------------------------------------------
seccion("7. models/metrics.json (si ya corriste models.py)")
try:
    import json
    from pathlib import Path

    ruta = M.DIR_MODELOS / "metrics.json"
    if ruta.exists():
        met = json.loads(ruta.read_text(encoding="utf-8"))
        claves = {"seed", "umbral_eca_pm25", "features", "mejor_modelo", "modelos"}
        check("metrics.json tiene las claves esperadas", claves <= set(met.keys()))
        check("seed == 96", met.get("seed") == 96, f"seed={met.get('seed')}")
        check("3 modelos reportados", len(met.get("modelos", {})) == 3,
              f"modelos={list(met.get('modelos', {}))}")
    else:
        check("metrics.json existe", False,
              "aún no lo generas: corre `uv run python src/models.py`", warn=True)
except Exception:  # noqa: BLE001
    check("lectura de metrics.json", False, "ver traceback")
    traceback.print_exc()

# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RESUMEN:  fallos={_fallos}  avisos={_avisos}")
if _fallos == 0:
    print("TODO OK ✔  (revisa los WARN si los hay)")
else:
    print("HAY FALLOS  — revisa las líneas [FAIL] de arriba.")
sys.exit(1 if _fallos else 0)