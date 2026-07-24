"""Serie temporal y pronóstico de PM2.5 (naive estacional, Holt-Winters, SARIMA)."""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.preprocessing import cargar_y_limpiar, SEED  # noqa: F401

# --- Constantes (mapa de parámetros: ver docs/mapa_parametros.md) ------------

OBJETIVO = "pm_25"
COL_FECHA = "fecha_hora"
COL_ESTACION = "estacion"
COL_IMPUTADO = f"{OBJETIVO}_imputado"

FREQ_DEFAULT = "MS"       # "MS" mensual, "W" semanal, "D" diaria
HORIZONTE = 6             # períodos futuros a pronosticar (rúbrica: >= 4)
PERIODOS_TEST = 12        # ventana de hold-out
UMBRAL_IMPUTADO = 0.5     # recorta la cola con > este % de PM2.5 imputado
TENDENCIA_HW = "add"
ESTACIONAL_HW = "add"

_ESTACIONALIDAD = {"MS": 12, "M": 12, "ME": 12, "W": 52, "D": 7}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RUTA_DATOS = BASE_DIR / "data" / "air_contamination.csv"
DIR_MODELOS = BASE_DIR / "models"
DIR_FIG = DIR_MODELOS / "figuras"
RUTA_METRICS = DIR_MODELOS / "forecast_metrics.json"

NAIVE = "naive_estacional"
HW = "holt_winters"
SARIMA = "sarima"


def estacionalidad(freq: str) -> int:
    return _ESTACIONALIDAD.get(freq, 12)


# --- Construcción de la serie -------------------------------------------------

def construir_serie(
    df: pd.DataFrame, estacion: str | None = None, freq: str = FREQ_DEFAULT,
    objetivo: str = OBJETIVO, recortar_imputados: bool = True,
    umbral_imputado: float = UMBRAL_IMPUTADO,
) -> pd.Series:
    """Serie regular del `objetivo` a frecuencia `freq`; recorta cola muy imputada."""
    d = df.copy()
    d[COL_FECHA] = pd.to_datetime(d[COL_FECHA])
    if estacion not in (None, "TODAS"):
        d = d[d[COL_ESTACION] == estacion]
        if d.empty:
            raise ValueError(f"No hay filas para la estación {estacion!r}.")
    d = d.set_index(COL_FECHA).sort_index()

    serie = d[objetivo].resample(freq).mean()

    if recortar_imputados and COL_IMPUTADO in d.columns and objetivo == OBJETIVO:
        frac_imp = d[COL_IMPUTADO].astype(float).resample(freq).mean()
        reales = frac_imp[frac_imp <= umbral_imputado]
        if len(reales):
            serie = serie.loc[: reales.index.max()]

    serie = serie.dropna().asfreq(freq)
    if serie.isna().any():
        serie = serie.interpolate(limit_direction="both")
    serie.name = f"{objetivo}__{estacion or 'TODAS'}"
    return serie


def dividir_serie(serie: pd.Series, periodos_test: int = PERIODOS_TEST):
    """Hold-out cronológico: los últimos `periodos_test` puntos como test."""
    n = len(serie)
    n_test = min(periodos_test, max(2, n // 5))
    if n - n_test < 4:
        raise ValueError(f"Serie demasiado corta (n={n}) para un hold-out fiable.")
    return serie.iloc[:-n_test], serie.iloc[-n_test:]


# --- Métricas ------------------------------------------------------------------

def _rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100.0)


def _mae(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def metricas(y_true, y_pred) -> dict:
    return {"mape": _mape(y_true, y_pred), "rmse": _rmse(y_true, y_pred), "mae": _mae(y_true, y_pred)}


# --- Modelos de pronóstico -------------------------------------------------------

def pred_naive_estacional(train: pd.Series, m: int, h: int) -> np.ndarray:
    """Baseline: cada período futuro = mismo período del último ciclo observado."""
    ultimo_ciclo = train.values[-m:] if len(train) >= m else train.values
    reps = int(np.ceil(h / len(ultimo_ciclo)))
    return np.tile(ultimo_ciclo, reps)[:h]


def ajustar_hw(train: pd.Series, m: int):
    """Holt-Winters aditivo; cae a no-estacional si falta historia."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    estacional = ESTACIONAL_HW if len(train) >= 2 * m else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            train, trend=TENDENCIA_HW, seasonal=estacional,
            seasonal_periods=m if estacional else None, initialization_method="estimated",
        )
        return modelo.fit()


def ajustar_sarima(train: pd.Series, m: int, orden=(1, 1, 1)):
    """SARIMA (1,1,1)(1,0,1)_m; cae a ARIMA no estacional si la serie es corta."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    seas = (1, 0, 1, m) if len(train) >= 2 * m else (0, 0, 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            train, order=orden, seasonal_order=seas,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        return modelo.fit(disp=False)


def _forecast(nombre: str, ajuste, h: int, m: int, train: pd.Series):
    """(yhat, lo, hi); lo/hi = None si el modelo no da intervalo."""
    if nombre == NAIVE:
        return pred_naive_estacional(train, m, h), None, None
    if nombre == HW:
        yhat = np.asarray(ajuste.forecast(h), float)
        return np.clip(yhat, 0, None), None, None
    res = ajuste.get_forecast(h)
    yhat = np.asarray(res.predicted_mean, float)
    ci = np.asarray(res.conf_int(alpha=0.20), float)  # 80% IC
    return np.clip(yhat, 0, None), np.clip(ci[:, 0], 0, None), np.clip(ci[:, 1], 0, None)


# --- Comparación de modelos ------------------------------------------------------

def comparar_modelos(serie: pd.Series, freq: str = FREQ_DEFAULT, periodos_test: int = PERIODOS_TEST):
    """Ajusta los 3 modelos, pronostica el test y calcula MAPE/RMSE/MAE."""
    m = estacionalidad(freq)
    train, test = dividir_serie(serie, periodos_test)
    h = len(test)

    ajustes = {NAIVE: None}
    try:
        ajustes[HW] = ajustar_hw(train, m)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Holt-Winters falló: {e}")
    try:
        ajustes[SARIMA] = ajustar_sarima(train, m)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"SARIMA falló: {e}")

    resultados = {}
    for nombre, ajuste in ajustes.items():
        yhat, lo, hi = _forecast(nombre, ajuste, h, m, train)
        met = metricas(test.values, yhat)
        resultados[nombre] = {"ajuste": ajuste, "yhat": pd.Series(yhat, index=test.index), **met}

    tabla = (
        pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk in ("mape", "rmse", "mae")}
                      for k, v in resultados.items()})
        .T.sort_values("mape")
    )
    mejor = tabla.index[0]
    return tabla, resultados, mejor, (train, test)


# --- Pronóstico final + orquestación ----------------------------------------------

def pronostico_final(serie: pd.Series, nombre_modelo: str, freq: str = FREQ_DEFAULT,
                     horizonte: int = HORIZONTE) -> pd.DataFrame:
    """Reajusta el modelo elegido sobre la serie completa y pronostica `horizonte` períodos."""
    m = estacionalidad(freq)
    if nombre_modelo == HW:
        ajuste = ajustar_hw(serie, m)
    elif nombre_modelo == SARIMA:
        ajuste = ajustar_sarima(serie, m)
    else:
        ajuste = None
    yhat, lo, hi = _forecast(nombre_modelo, ajuste, horizonte, m, serie)

    idx_fut = pd.date_range(serie.index[-1], periods=horizonte + 1, freq=freq)[1:]
    out = pd.DataFrame({"yhat": yhat}, index=idx_fut)
    if lo is not None:
        out["lo"], out["hi"] = lo, hi
    return out


def serie_y_pronostico(df, estacion=None, freq=FREQ_DEFAULT, periodos_test=PERIODOS_TEST,
                       horizonte=HORIZONTE, recortar_imputados=True):
    """Todo-en-uno: serie -> comparar -> mejor -> futuro."""
    serie = construir_serie(df, estacion=estacion, freq=freq, recortar_imputados=recortar_imputados)
    tabla, resultados, mejor, (train, test) = comparar_modelos(serie, freq, periodos_test)
    futuro = pronostico_final(serie, mejor, freq, horizonte)
    return {
        "serie": serie, "tabla": tabla, "resultados": resultados, "mejor": mejor,
        "train": train, "test": test, "futuro": futuro,
        "config": {"estacion": estacion or "TODAS", "freq": freq, "estacionalidad": estacionalidad(freq),
                   "periodos_test": len(test), "horizonte": horizonte},
    }


def guardar_metrics_json(paquete: dict, ruta: Path = RUTA_METRICS) -> Path:
    """Persiste métricas auditables (fuente de verdad del Panel 3 y el Reporte)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla, cfg = paquete["tabla"], paquete["config"]
    doc = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "objetivo": OBJETIVO, **cfg,
        "mejor_modelo": paquete["mejor"],
        "n_serie": int(len(paquete["serie"])),
        "n_train": int(len(paquete["train"])),
        "n_test": int(len(paquete["test"])),
        "modelos": {
            nombre: {"mape": float(row.mape), "rmse": float(row.rmse), "mae": float(row.mae)}
            for nombre, row in tabla.iterrows()
        },
        "pronostico_futuro": [
            {"fecha": ts.strftime("%Y-%m-%d"), "yhat": float(r.yhat),
             "lo": (float(r.lo) if "lo" in paquete["futuro"].columns else None),
             "hi": (float(r.hi) if "hi" in paquete["futuro"].columns else None)}
            for ts, r in paquete["futuro"].iterrows()
        ],
    }
    ruta.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def entrenar_y_evaluar_todo(df=None, estacion=None, freq=FREQ_DEFAULT,
                            periodos_test=PERIODOS_TEST, horizonte=HORIZONTE, guardar=True):
    """Corre todo y persiste métricas (JSON); orquestador equivalente al de models.py."""
    if df is None:
        df = cargar_y_limpiar(str(RUTA_DATOS))
    paquete = serie_y_pronostico(df, estacion, freq, periodos_test, horizonte)
    if guardar:
        guardar_metrics_json(paquete)
    return paquete


if __name__ == "__main__":
    from application.graficos import graficar

    print(">> Forecasting de PM2.5 (Lima agregada)")
    paquete = entrenar_y_evaluar_todo()
    graficar(paquete, DIR_FIG / f"forecast_{paquete['config']['estacion'].replace(' ', '_')}.png")

    print("\nComparación de modelos (ordenada por MAPE):")
    print(paquete["tabla"].round(3).to_string())
    print(f"\nMejor modelo: {paquete['mejor']}")
    print(f"\nPronóstico {paquete['config']['horizonte']} períodos:")
    print(paquete["futuro"].round(2).to_string())
    print(f"\nMétricas guardadas en: {RUTA_METRICS}")
