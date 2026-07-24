"""
verificar_rol_c.py — smoke test del pipeline de series (Rol C).

Diagnóstico rápido PASS / WARN / FAIL del entorno y del flujo de forecast.py, sin
esperar la corrida completa de todas las estaciones. Análogo a verificar_rol_b.py.

    uv run python verificar_rol_c.py      # o:  python verificar_rol_c.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_estado = {OK: 0, WARN: 0, FAIL: 0}


def chk(nombre, estado, detalle=""):
    _estado[estado] += 1
    marca = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}[estado]
    print(f"{marca} [{estado}] {nombre}" + (f" — {detalle}" if detalle else ""))


print("=== verificar_rol_c: pipeline de series temporales ===\n")

# 1. Dependencias
try:
    import numpy, pandas, statsmodels, sklearn  # noqa
    chk("dependencias base", OK, f"numpy {numpy.__version__}, pandas {pandas.__version__}, "
                                 f"statsmodels {statsmodels.__version__}")
except Exception as e:
    chk("dependencias base", FAIL, str(e))
    sys.exit(1)

# 2. Import de módulos del proyecto
try:
    import core.forecast as F
    from core.preprocessing import cargar_y_limpiar, SEED
    chk("import forecast + preprocessing", OK, f"SEED={SEED}, HORIZONTE={F.HORIZONTE}")
except Exception as e:
    chk("import forecast + preprocessing", FAIL, str(e))
    sys.exit(1)

# 3. Datos disponibles
df = None
try:
    df = cargar_y_limpiar(str(F.RUTA_DATOS))
    faltan = [c for c in [F.COL_FECHA, F.OBJETIVO, F.COL_ESTACION, F.COL_IMPUTADO] if c not in df.columns]
    if faltan:
        chk("dataset limpio", FAIL, f"faltan columnas: {faltan}")
    else:
        chk("dataset limpio", OK, f"{len(df):,} filas, {df['estacion'].nunique()} estaciones")
except Exception as e:
    chk("dataset limpio", WARN, f"no se pudo cargar ({e}). Corre primero preprocessing.py")

# 4. Construcción de serie
serie = None
if df is not None:
    try:
        serie = F.construir_serie(df, estacion=None, freq=F.FREQ_DEFAULT)
        assert serie.notna().all() and len(serie) >= 24
        chk("construir_serie (Lima, mensual)", OK,
            f"n={len(serie)}, {serie.index.min().date()}→{serie.index.max().date()}")
    except Exception as e:
        chk("construir_serie", FAIL, str(e))

# 5. Comparación de modelos + métricas
if serie is not None:
    try:
        tabla, resultados, mejor, (tr, te) = F.comparar_modelos(serie)
        assert set(resultados) == {F.NAIVE, F.HW, F.SARIMA}
        assert mejor in resultados
        mape = tabla.loc[mejor, "mape"]
        assert mape > 0 and "rmse" in tabla.columns
        chk("comparar_modelos (3 modelos, MAPE/RMSE)", OK,
            f"mejor={mejor}, MAPE={mape:.2f}%, train={len(tr)}, test={len(te)}")
    except Exception as e:
        chk("comparar_modelos", FAIL, str(e))
        mejor = None

# 6. Pronóstico >= 4 períodos
if serie is not None and 'mejor' in dir() and mejor:
    try:
        fut = F.pronostico_final(serie, mejor, horizonte=F.HORIZONTE)
        assert len(fut) == F.HORIZONTE >= 4 and "yhat" in fut.columns
        chk("pronostico_final (≥4 períodos)", OK, f"{len(fut)} períodos, futuro hasta {fut.index[-1].date()}")
    except Exception as e:
        chk("pronostico_final", FAIL, str(e))

# 7. Panel importable
try:
    import application.panel_forecast as panel_forecast  # noqa
    assert hasattr(panel_forecast, "render")
    chk("panel_forecast.render importable", OK)
except Exception as e:
    chk("panel_forecast", WARN, str(e))

print(f"\nResumen: {_estado[OK]} PASS · {_estado[WARN]} WARN · {_estado[FAIL]} FAIL")
sys.exit(1 if _estado[FAIL] else 0)
