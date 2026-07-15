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


## Roles y responsabilidades

| Rol | Responsabilidad | Entregables clave |
|-|-|-|
| **A — Datos y EDA** | Limpieza, imputación, EDA, clustering K-means | `src/preprocessing.py`, `notebooks/01_eda.ipynb`, Panel 1 |
| **B — Modelado y XAI** | Etiqueta binaria, RF vs XGBoost, matriz de confusión, SMOTE, SHAP | `src/models.py`, `notebooks/02_modeling.ipynb`, Panel 2 |
| **C — Series e infra** | Pronóstico, deploy, integración de los 4 paneles | `src/forecast.py`, `app.py` |
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

# 3. (Rol A) Notebook de EDA + clustering
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb

# 4. (Rol B) Notebook de modelado + XAI
uv run jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb --output 02_modeling.ipynb

# 5. Demo aislada del Panel 2 (predictivo) sin esperar a la app integrada
uv run streamlit run src/panel_predictivo.py
```

> El dataset limpio (`data/*.parquet`) y los binarios de modelo (`models/*.pkl`, figuras)
> **no** se versionan: se regeneran en segundos con los pasos 1–2. Sí se versiona
> `models/metrics.json` para auditar métricas sin reentrenar.

## Estructura

```
.
├── data/
│   └── air_contamination.csv          # crudo SENAMHI (versionado)
├── src/
│   ├── preprocessing.py               # Rol A: carga, limpieza, imputación, constantes
│   ├── models.py                      # Rol B: etiqueta, RF vs XGBoost, SMOTE, métricas, SHAP
│   └── panel_predictivo.py            # Rol B: contenido del Panel 2 (importable por app.py)
├── notebooks/
│   ├── 01_eda.ipynb                   # Rol A: EDA + clustering
│   └── 02_modeling.ipynb              # Rol B: modelado + interpretabilidad
├── models/
│   └── metrics.json                   # métricas auditables (versionado)
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