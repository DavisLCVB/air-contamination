# Contaminación del aire en Lima Metropolitana

**"Dos Limas, un mismo cielo: clustering y predicción de calidad del aire en Lima Metropolitana"**

Pipeline CRISP-DM sobre datos horarios de calidad del aire de SENAMHI (10 estaciones,
2014–2020). Clasifica horas de **alta contaminación por PM2.5** (etiqueta ECA > 50 µg/m³)
a partir del resto de contaminantes, con manejo de desbalance e interpretabilidad (SHAP).

## Autores

| Apellidos y Nombres | Correo | Código | Rol |
|-|-|-|-|
| Cartagena Valera Brush, Davis Leonardo | davis.cartagena@unmsm.edu.pe | 22200193 | A — Datos y EDA |
| Lavado Torres, Gianmarco Gabriel | gianmarco.lavado@unmsm.edu.pe | 22200025 | B — Modelado y XAI |
| Chilon Tintaya Monica Isabel |  |  | C — Series e infra |


## Roles y responsabilidades

| Rol | Responsabilidad | Entregables clave |
|-|-|-|
| **A — Datos y EDA** | Limpieza, imputación, EDA, clustering K-means | `src/preprocessing.py`, `notebooks/01_eda.ipynb`, Panel 1 |
| **B — Modelado y XAI** | Etiqueta binaria, RF vs XGBoost, matriz de confusión, SMOTE, SHAP | `src/models.py`, `notebooks/02_modeling.ipynb`, Panel 2 |
| **C — Series e infra** | Pronóstico, deploy, integración de los 4 paneles | `src/forecast.py`, `src/panel_forecast.py`, `notebooks/03_forecasting.qmd`, `app.py` |
| **D — CRUD y reporte** | CRUD de consultas, Reporte PDF | Panel 4, Reporte PDF |

## Requisitos

- Python **≥ 3.12**
- [`uv`](https://docs.astral.sh/uv/) para gestión de entorno y dependencias.

## Instalación

```bash
uv sync                 # crea el entorno y resuelve dependencias desde uv.lock
# opcional (complemento de interpretabilidad LIME):
# uv sync --extra lime
```

## Reproducir el proyecto desde cero

Todo el flujo es reproducible sin pasos manuales. La semilla oficial es `SEED = 96`
(definida en `src/preprocessing.py` e importada por el resto de módulos), de modo que
cualquier split, muestreo o modelo aleatorio produce **los mismos resultados**.

```bash
# 1. Regenerar el dataset limpio (Rol A). Genera data/air_contamination_clean.parquet
uv run python src/preprocessing.py

# 2. Entrenar modelos, evaluar y generar artefactos de XAI (Rol B).
#    Produce models/rf.pkl, models/xgb.pkl, models/metrics.json y figuras.
uv run python src/models.py

# 3. Pronóstico de series temporales (Rol C).
#    Produce models/forecast_metrics.json y la figura del pronóstico.
uv run python src/forecast.py

# 4. (Rol A) Notebook de EDA + clustering
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb

# 5. (Rol B) Notebook de modelado + XAI
uv run jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb --output 02_modeling.ipynb

# 6. (Rol C) Notebook de series temporales
uv run quarto render notebooks/03_forecasting.qmd

# 7. Demo aislada del Panel 2 (predictivo) sin esperar a la app integrada
uv run streamlit run src/panel_predictivo.py

# 8. App integrada — los 4 paneles en un solo dashboard
uv run streamlit run app.py
```

> El dataset limpio (`data/*.parquet`) y los binarios de modelo (`models/*.pkl`, figuras)
> **no** se versionan: se regeneran en segundos con los pasos 1–2. Sí se versionan
> `models/metrics.json` y `models/forecast_metrics.json` para auditar métricas sin reentrenar.

## Estructura

```
.
├── app.py                             # Rol C: dashboard integrado (4 paneles)
├── data/
│   └── air_contamination.csv          # crudo SENAMHI (versionado)
├── src/
│   ├── preprocessing.py               # Rol A: carga, limpieza, imputación, constantes
│   ├── models.py                      # Rol B: etiqueta, RF vs XGBoost, SMOTE, métricas, SHAP
│   ├── panel_predictivo.py            # Rol B: contenido del Panel 2 (importable por app.py)
│   ├── forecast.py                    # Rol C: serie temporal, modelos, MAPE/RMSE
│   └── panel_forecast.py              # Rol C: contenido del Panel 3 (importable por app.py)
├── notebooks/
│   ├── 01_eda.ipynb                   # Rol A: EDA + clustering
│   ├── 02_modeling.ipynb              # Rol B: modelado + interpretabilidad
│   └── 03_forecasting.qmd             # Rol C: series temporales
├── models/
│   ├── metrics.json                   # métricas de clasificación (versionado)
│   └── forecast_metrics.json          # métricas de pronóstico (versionado)
├── requirements.txt                   # Rol C: dependencias para el deploy
├── pyproject.toml
└── uv.lock
```

## Decisiones de modelado (Rol B) — resumen

- **Etiqueta:** `y = (pm_25 > 50)` (constante `ECA_PM25 = 50`, cambiable en un solo lugar).
- **Features:** los 5 contaminantes restantes (`pm_10, so2, no2, o3, co`). Se excluyen
  `pm_25` y `pm_25_imputado` para evitar fuga de información.
- **Etiqueta real:** se entrena y evalúa solo sobre filas con PM2.5 **medido**
  (`pm_25_imputado == False`), para no medir la calidad de la imputación en vez de la
  predicción.
- **Desbalance (~92/8):** se compara `class_weight='balanced'` vs **SMOTE**, reportando
  el efecto en el recall de la clase minoritaria.
- **Sin escalado:** RF y XGBoost son invariantes a escala (a diferencia del clustering
  de Rol A, que sí usa `StandardScaler`).
- **XAI:** SHAP `TreeExplainer` con `summary_plot` (global) y `force_plot` (local).

## Decisiones de modelado (Rol C) — resumen

- **Objetivo:** pronosticar `pm_25` como **serie mensual** (período estacional `m = 12`:
  el "invierno limeño").
- **Modelos:** *naive estacional* (baseline), *Holt-Winters* y *SARIMA*. Se elige el de
  menor MAPE en un hold-out cronológico y se reajusta sobre toda la serie para el futuro.
- **Métricas:** **MAPE** y **RMSE** sobre los últimos 12 meses (hold-out **sin barajar**).
- **Cola imputada:** se recorta el tramo final con PM2.5 mayormente imputado, para no
  pronosticar sobre climatología.
- **Sin fuga:** el pronóstico usa solo el pasado de la propia serie. Cifras auditables en
  `models/forecast_metrics.json`.

