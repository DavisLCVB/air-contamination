# Paneles del dashboard — "Dos Limas, un mismo cielo"

Este documento describe, panel por panel, qué muestra cada gráfico y control de
`app.py` y por qué está ahí. Sirve como referencia para el reporte, la presentación
y para cualquiera que necesite modificar un panel sin releer todo el código.

Los 4 paneles son pestañas independientes en `app.py`; cada una llama a
`render(df)` del módulo correspondiente en `src/`.

---

## Panel 1 · EDA & Clustering

**Módulo:** `src/application/panel_eda.py`

Perfila la contaminación por zona y agrupa las estaciones según qué tan
contaminadas están, para sustentar con datos la tesis de las "Dos Limas": no
toda la ciudad respira el mismo aire.

### Resumen general (4 métricas)

| Métrica | Qué muestra |
|---|---|
| Filas | Total de registros horarios tras la limpieza (`preprocessing.limpiar_e_imputar`). |
| Estaciones | Número de estaciones SENAMHI presentes en el dataset. |
| Rango | Años cubiertos por la serie (min–max de `fecha_hora`). |
| PM2.5 imputado | % de horas donde el PM2.5 no llegó medido y se rellenó (interpolación de huecos cortos o climatología horaria), no un dato real. |

### PM2.5 promedio por estación (gráfico de barras)

Promedio histórico de PM2.5 por estación, ordenado de mayor a menor. La línea
punteada marca el Estándar de Calidad Ambiental (ECA = 50 µg/m³); las barras se
pintan de rojo solo si el promedio de esa estación lo supera. Con los datos
actuales **ninguna estación cruza ese umbral** (la más alta es ATE, ~39.4
µg/m³) — el caption debajo del gráfico se calcula en vivo sobre los datos, así
que si en algún reentrenamiento futuro alguna estación sí lo supera, el texto
cambia solo. La desigualdad entre zonas sigue siendo evidente en la brecha
entre estaciones, aunque ninguna supere el ECA en promedio.

### Correlación entre contaminantes (heatmap)

Matriz de correlación de Pearson entre los contaminantes disponibles
(`pm_10, pm_25, so2, no2, o3, co`). Valores cercanos a 1 (rojo) indican
contaminantes que suben y bajan juntos —típico de fuentes compartidas como
tráfico vehicular o quema—; cercanos a 0 (azul) indican que no están
relacionados.

### Clustering de estaciones (K-means)

- **Fila de control + métricas (3 columnas, lado a lado):**
  - **Slider "Número de clusters (k)"** (2 a 6, default 2): número de grupos
    para K-means. k=2 separa naturalmente alta vs. baja contaminación.
  - **Métrica "Inercia"**: del modelo para el k actual (útil para el método del
    codo al elegir k).
  - **Métrica "Silueta"**: qué tan bien separado está cada punto de los
    clusters vecinos para el k actual (−1 a 1, más alto es mejor).
  - Van en la misma fila a propósito: al mover el slider, inercia y silueta se
    actualizan sin tener que bajar a buscarlas.
- **Caption de advertencia** (debajo de esa fila, siempre visible): la silueta
  se calcula sobre las 10 estaciones (una fila por estación), no sobre las
  ~515,000 filas hora-estación del notebook de EDA — con tan pocos puntos,
  subir k deja clusters de 1-2 miembros y la métrica se vuelve ruidosa. **No es
  comparable** con la silueta del notebook, aunque en la práctica ambas
  coinciden en que k=2 es el mejor valor.
- **Gráfico de dispersión (PCA):** cada punto es una estación coloreada por su
  cluster. Los ejes "PCA 1" / "PCA 2" son solo una proyección 2D para poder
  dibujar — el agrupamiento real se calcula sobre todos los contaminantes
  estandarizados a la vez, no sobre estos dos ejes.
- **Tabla de perfil por cluster:** promedio histórico de cada contaminante por
  estación, ordenada por cluster asignado.
- **Expander "Detalles técnicos del clustering":** unidad de clustering
  (estación = promedio de sus contaminantes), estandarización (`StandardScaler`
  antes de K-means, para que ningún contaminante domine solo por su escala),
  semilla (`SEED = 96`, reproducible) y nota sobre la proyección PCA.

---

## Panel 2 · Predicción de alta contaminación

**Módulo:** `src/application/panel_predictivo.py`

Clasifica cada hora como *alta contaminación* cuando `PM2.5 > 50 µg/m³` (umbral
ECA), usando el resto de contaminantes (`pm_10, so2, no2, o3, co`), `hora`,
`mes`, `estacion` y la media móvil de las 3h **anteriores** de cada
contaminante (rezago, por estación, ver `core/models.py::_agregar_rezago`)
como variables. `estacion` se codifica con one-hot dentro del propio Pipeline
del modelo (`core/models.py::_preprocesador`). `pm_25` se excluye de las
features para evitar fuga de la variable objetivo, y el modelo se entrena
solo con PM2.5 **medido** (no imputado). La clase "alta" es minoritaria
(~7.9% de las horas), de ahí el manejo explícito del desbalance
(SMOTE / `class_weight`).

### Comparación de modelos (tabla)

Compara `rf_classweight`, `xgb_scaleposw` y `rf_smote` en F1 / Recall /
Precision de la clase "alta", ROC-AUC y Accuracy. El modelo ganador se elige
por **F1 de la clase minoritaria** (no por accuracy), porque el costo de no
avisar una hora contaminada (falso negativo) es mayor que una falsa alarma
(falso positivo).

### Matriz de confusión (imagen)

Filas = clase real, columnas = clase predicha, para el modelo ganador. La
diagonal son los aciertos; fuera de la diagonal, los falsos positivos y falsos
negativos.

### Importancia global — SHAP summary (imagen)

Cada punto es una hora del set de test. Arriba, los contaminantes que más
mueven la predicción en general; el color indica si ese contaminante estaba
alto (rojo) o bajo (azul) en esa hora.

### Explicación local — SHAP force plot (imagen)

Cómo se arma la predicción para un caso puntual: cada contaminante empuja la
probabilidad hacia "alta contaminación" (rojo) o hacia "baja" (azul) desde un
valor base, hasta llegar a la probabilidad final del modelo.

### Probar una predicción (controles interactivos)

- **Slider "Umbral de decisión"** (0.05–0.95, default = umbral de
  `metrics.json`): bajarlo sube el recall de "alta" a costa de más falsos
  positivos.
- **8 campos** (`pm_10, so2, no2, o3, co, hora, mes` como numéricos y
  `estacion` como desplegable): lectura de los otros contaminantes, la hora,
  el mes y la estación de esa hora. El formulario NO pide el rezago (no hay
  historia real de esa estación disponible en un input manual): internamente
  `predecir_desde_entrada` asume "sin tendencia" y usa el mismo valor
  ingresado como su propio rezago.
- **Botón "Predecir":** corre el modelo ganador sobre los valores ingresados y
  muestra la etiqueta (alta/baja contaminación), la probabilidad y el umbral
  usado. El resultado queda en `st.session_state["ultima_prediccion"]` para que
  el futuro Panel 4 (CRUD) lo pueda persistir.

Si `models/*.pkl` no existe en disco, el panel entrena en caliente una sola vez
(cacheado) y lo avisa con una nota al pie.

---

## Panel 3 · Serie temporal y pronóstico

**Módulo:** `src/application/panel_forecast.py` (lógica de series en `src/core/forecast.py`)

Pronostica PM2.5 a futuro (≥ 4 períodos) comparando tres modelos —naive
estacional, Holt-Winters y SARIMA— y eligiendo el de menor MAPE en un
hold-out cronológico (sin barajar, para no filtrar futuro hacia el pasado).

### Controles

| Control | Qué hace |
|---|---|
| Selectbox "Estación" | Estación a pronosticar. "TODAS" = promedio de Lima. Comparar ATE/PUENTE PIEDRA (alta) vs. CAMPO DE MARTE (baja) ilustra las "Dos Limas". |
| Radio "Frecuencia" | Intervalo al que se agrega el PM2.5 antes de modelar (mensual/semanal/diaria) — dato horario es demasiado ruidoso para pronosticar directo. |
| Slider "Horizonte" | Cuántos períodos futuros se proyectan, en la unidad elegida en "Frecuencia". |
| Slider "Ventana de test" | Cuántos períodos finales de la serie se separan como hold-out para medir MAPE/RMSE antes de pronosticar el futuro. |
| Checkbox "Recortar cola" | Evita pronosticar sobre tramos donde el PM2.5 es casi todo climatología (relleno estimado, no medido). No afecta estaciones con dato real hasta el final. |

### Métricas (4 tarjetas)

Mejor modelo, MAPE (test), RMSE (test) y puntos de la serie. MAPE es el error
porcentual promedio en el hold-out (más bajo es mejor); RMSE está en las
mismas unidades que PM2.5.

### Gráfico de pronóstico

Serie histórica de PM2.5 y, al final, el tramo de test (real vs. predicho por
el modelo ganador) seguido del pronóstico hacia adelante.

### Comparación de modelos (tabla)

MAPE %, RMSE y MAE de los tres modelos, ordenada por MAPE; el valor resaltado
en verde es el mínimo de cada métrica.

### Pronóstico — próximos N períodos (tabla)

Valores proyectados por el modelo ganador, reajustado sobre toda la serie
disponible (no solo sobre el tramo de entrenamiento del hold-out).

### Expander "Detalles técnicos del pronóstico"

Objetivo (`pm_25` a la frecuencia elegida), split cronológico, métricas
usadas, el baseline honesto (naive estacional: copiar el último ciclo — un
modelo solo se justifica si le gana), la ausencia de fuga (el pronóstico usa
solo el pasado de la propia serie) y dónde están las cifras auditables
(`models/forecast_metrics.json`).

---

## Panel 4 · CRUD de consultas & Reporte

**Módulo:** `src/application/panel_crud.py`

CRUD completo sobre SQLite (`data/consultas.db`, no versionado — se crea solo
al primer uso) que registra "consultas de predicción": cada una guarda los
datos de entrada (contaminantes) junto con la predicción devuelta por el
modelo del Panel 2, más nombre, correo, tipo de consulta, mensaje y timestamp
automático.

### Predictor con respaldo en 3 niveles

Al cargar el panel, `_cargar_predictor()` intenta en orden:

1. Reutilizar `predecir_desde_entrada(modelo, valores)` de `panel_predictivo.py`
   (vía preferida, no depende del nombre de archivo del modelo).
2. Cargar directamente el modelo serializado con `joblib` si el paso 1 falla.
3. Si ningún modelo existe todavía en disco, usar un predictor heurístico de
   respaldo (**no** es el modelo entrenado) para que el CRUD siga siendo
   demostrable en la presentación. El panel avisa con una nota (`⚠️`) cuando
   está en este modo.

### Pestañas (Crear / Listar / Editar / Eliminar)

- **Crear:** formulario con nombre (obligatorio), correo, tipo de consulta,
  mensaje y las 8 variables del modelo (`pm_10, so2, no2, o3, co, hora, mes` y
  `estacion` como desplegable). Al enviar, corre el predictor activo y guarda
  entrada + predicción en SQLite.
- **Listar:** tabla con todas las consultas (más recientes primero) y botón
  para descargar el historial completo en CSV.
- **Editar:** actualiza nombre, correo, tipo de consulta y mensaje de una
  consulta existente por id (no re-corre la predicción).
- **Eliminar:** borra una consulta por id, con checkbox de confirmación
  explícita antes de habilitar el botón.
