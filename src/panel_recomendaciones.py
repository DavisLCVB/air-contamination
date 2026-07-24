"""
panel_recomendaciones.py — Panel 5 (Recomendaciones de intervención) del dashboard.

Sintetiza el clustering del Panel 1 (severidad: ¿la estación está en el grupo de alta
contaminación?) y la lógica de series del Panel 3 (trayectoria: ¿su PM2.5 mensual mejora
por sí solo?) en una lista de estaciones prioritarias para intervención. Expone
`render(df=None)` igual que los demás paneles (demo aislada:
uv run streamlit run src/panel_recomendaciones.py).
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import forecast as F
import theme
from panel_eda import ECA_PM25_ANUAL, _clusters, _perfil_por_estacion
from preprocessing import cargar_y_limpiar

N_MIN = 24  # meses mínimos para confiar en la pendiente de tendencia

# Zona muerta para la pendiente: con series de ~5-6 años, una pendiente dentro de
# ±0.01 µg/m³/mes es indistinguible del ruido natural de la serie. Se trata como
# "sin cambio confirmado" (no como mejora) para no penalizar por una pendiente
# negativa que en realidad es prácticamente plana.
UMBRAL_MEJORA = -0.01  # µg/m³ por mes

# Coordenadas aproximadas (centro del distrito/localidad, fuente: Wikipedia/geodatos
# públicos) — referenciales para ubicar la estación en el mapa, no son las coordenadas
# exactas del instrumento SENAMHI (no publicadas).
_COORDENADAS = {
    "ATE": (-12.0103, -76.8700),
    "CAMPO DE MARTE": (-12.0681, -77.0419),
    "SAN BORJA": (-12.1000, -77.0170),
    "SANTA ANITA": (-12.0432, -76.9631),
    "VILLA MARIA DEL TRIUNFO": (-12.1570, -76.9310),
    "HUACHIPA": (-11.9988, -76.9307),
    "SAN JUAN DE LURIGANCHO": (-12.0330, -77.0170),
    "SAN MARTIN DE PORRES": (-12.0278, -77.0433),
    "CARABAYLLO": (-11.8500, -77.0330),
    "PUENTE PIEDRA": (-11.8750, -77.0653),
}


@st.cache_data(show_spinner=False)
def _cargar_df():
    return cargar_y_limpiar(str(F.RUTA_DATOS))


def _pendiente(serie: pd.Series) -> float:
    """Pendiente OLS (µg/m³ por mes) de una serie mensual; NaN si hay muy pocos puntos."""
    if len(serie) < N_MIN:
        return float("nan")
    x = np.arange(len(serie))
    return float(np.polyfit(x, serie.values, 1)[0])


def _texto_tendencia(v: float) -> str:
    """Etiqueta legible de la pendiente, marcando la zona muerta como 'estable'."""
    if pd.isna(v):
        return "—"
    banda = abs(UMBRAL_MEJORA)
    if v > banda:
        return f"▲ {v:+.3f} µg/m³/mes"
    if v < -banda:
        return f"▼ {v:+.3f} µg/m³/mes"
    return f"→ {v:+.3f} µg/m³/mes (estable)"


@st.cache_data(show_spinner=True)
def _tabla_prioridad() -> pd.DataFrame:
    """Une severidad (cluster K-means, k=2) y trayectoria (pendiente mensual) por estación."""
    df = _cargar_df()
    perfil = _perfil_por_estacion(df)
    labels, _coords, _inercia, _silueta = _clusters(perfil, 2)

    medias_cluster = perfil["pm_25"].groupby(labels).mean()
    label_alta = medias_cluster.idxmax()

    filas = []
    for i, estacion in enumerate(perfil.index):
        serie = F.construir_serie(df, estacion=estacion, freq="MS")
        filas.append({
            "estacion": estacion,
            "pm25": float(perfil.loc[estacion, "pm_25"]),
            "cluster_alta": bool(labels[i] == label_alta),
            "pendiente": _pendiente(serie),
            "n_meses": len(serie),
        })

    tabla = pd.DataFrame(filas)
    tabla["criterio_severidad"] = tabla["cluster_alta"]
    tabla["criterio_trayectoria"] = tabla["pendiente"] >= UMBRAL_MEJORA
    tabla["prioridad"] = tabla["criterio_severidad"] & tabla["criterio_trayectoria"]
    return tabla.sort_values(["prioridad", "pm25"], ascending=[False, False]).reset_index(drop=True)


def render(df=None):
    """Dibuja el Panel 5: cruza severidad (Panel 1) y trayectoria (Panel 3) en una recomendación."""
    st.subheader(":material/recommend: Panel 5 · Recomendaciones de intervención")
    st.caption(
        "Cruza el cluster de alta contaminación (Panel 1) con la tendencia mensual de "
        "PM2.5 (misma lógica de series del Panel 3) para señalar qué estaciones priorizar."
    )

    p = theme.paleta()
    tabla = _tabla_prioridad()

    with st.expander(":material/rule: Metodología: cómo se calcula la prioridad", expanded=False):
        st.caption(
            "**Criterio A — severidad:** la estación cae en el cluster de alta "
            "contaminación (K-means, k=2, igual que el Panel 1)."
        )
        st.caption(
            f"**Criterio B — trayectoria:** la tendencia mensual de PM2.5 no muestra una "
            f"mejora confirmada (pendiente ≥ {UMBRAL_MEJORA:+.2f} µg/m³/mes, regresión lineal "
            "sobre la serie mensual del Panel 3). Un margen de "
            f"±{abs(UMBRAL_MEJORA):.2f} µg/m³/mes se trata como 'sin cambio confirmado', no "
            "como mejora, dado el ruido natural de una serie de pocos años."
        )
        st.caption(
            "Una estación es prioritaria solo si cumple **ambos** criterios a la vez — sin "
            "mezclar señales en un puntaje ponderado poco transparente."
        )

    n_prioridad = int(tabla["prioridad"].sum())
    n_cluster_alta = int(tabla["cluster_alta"].sum())
    pm25_prioridad = tabla.loc[tabla["prioridad"], "pm25"].mean() if n_prioridad else float("nan")

    with st.container(horizontal=True):
        st.metric("Estaciones prioritarias", n_prioridad, border=True)
        st.metric("En cluster de alta contaminación", n_cluster_alta, border=True)
        st.metric("PM2.5 promedio (prioritarias)",
                  f"{pm25_prioridad:.1f} µg/m³" if n_prioridad else "—", border=True)

    with st.container(border=True):
        st.subheader(":material/location_on: Explorar por estación")
        col_resumen, col_mapa = st.columns([1, 1])

        with col_resumen:
            opciones = tabla["estacion"].tolist()
            seleccion = st.selectbox("Estación", opciones, index=0, key="estacion_detalle")
            fila = tabla.loc[tabla["estacion"] == seleccion].iloc[0]

            st.metric("PM2.5 promedio", f"{fila['pm25']:.1f} µg/m³", border=True)
            st.metric("Cluster",
                      "Alta contaminación" if fila["cluster_alta"] else "Baja contaminación",
                      border=True)
            st.metric("Tendencia mensual", _texto_tendencia(fila["pendiente"]), border=True)

            if fila["criterio_severidad"]:
                texto_severidad = "está en el **cluster de alta contaminación** (Panel 1)."
            else:
                texto_severidad = "está en el **cluster de baja contaminación** (Panel 1)."
            if pd.isna(fila["pendiente"]):
                texto_trayectoria = "no hay suficientes meses de datos para evaluar su tendencia."
            elif fila["criterio_trayectoria"]:
                texto_trayectoria = (
                    "su PM2.5 mensual **no muestra una mejora confirmada** "
                    f"(pendiente {fila['pendiente']:+.3f} µg/m³/mes)."
                )
            else:
                texto_trayectoria = (
                    "su PM2.5 mensual **baja de forma confirmada** por sí solo "
                    f"(pendiente {fila['pendiente']:+.3f} µg/m³/mes, más allá del margen de "
                    f"±{abs(UMBRAL_MEJORA):.2f})."
                )
            st.caption(f":material/rule: Criterio A: {seleccion} {texto_severidad}")
            st.caption(f":material/trending_up: Criterio B: {texto_trayectoria}")

            if fila["prioridad"]:
                st.success(
                    f"**{seleccion} es una recomendación de tratamiento**: cumple ambos criterios a la vez.",
                    icon=":material/priority_high:",
                )
            else:
                st.warning(
                    f"**{seleccion} no es una recomendación de tratamiento** por ahora: "
                    "no cumple ambos criterios a la vez.",
                    icon=":material/info:",
                )

        with col_mapa:
            coords = _COORDENADAS.get(seleccion)
            if coords:
                st.map(pd.DataFrame({"lat": [coords[0]], "lon": [coords[1]]}), zoom=12, size=250)
                st.caption(
                    "Ubicación aproximada (centro del distrito/localidad, coordenadas "
                    "públicas de referencia — no la posición exacta del instrumento SENAMHI)."
                )
            else:
                st.info("No hay coordenadas registradas para esta estación.")

    with st.container(border=True):
        st.subheader(":material/table_chart: Ranking completo de estaciones")
        vista = tabla.copy()
        vista["Cluster"] = vista["cluster_alta"].map({True: "Alta", False: "Baja"})
        vista["Tendencia"] = vista["pendiente"].apply(_texto_tendencia)
        vista = vista.rename(columns={"estacion": "Estación", "pm25": "PM2.5 promedio",
                                        "prioridad": "Prioridad"})
        st.dataframe(
            vista[["Estación", "PM2.5 promedio", "Cluster", "Tendencia", "Prioridad"]],
            column_config={
                "PM2.5 promedio": st.column_config.NumberColumn(format="%.1f µg/m³"),
                "Prioridad": st.column_config.CheckboxColumn(),
            },
            hide_index=True, width="stretch",
        )
        st.caption("Ordenado primero por prioridad y luego por PM2.5 promedio, de mayor a menor.")

    with st.container(border=True):
        st.subheader(":material/scatter_plot: Severidad vs. tendencia")

        datos_chart = tabla.copy()
        datos_chart["estado"] = datos_chart["prioridad"].map(
            {True: "Prioritaria", False: "No prioritaria"})

        puntos = alt.Chart(datos_chart).mark_circle(
            size=200, opacity=0.9, stroke=p["TEXTO"], strokeWidth=0.8
        ).encode(
            x=alt.X("pm25:Q", title="PM2.5 promedio (µg/m³)"),
            y=alt.Y("pendiente:Q", title="Tendencia (µg/m³ por mes)"),
            color=alt.Color("estado:N", title=None,
                             scale=alt.Scale(domain=["Prioritaria", "No prioritaria"],
                                             range=[p["ROJO"], p["AZUL"]])),
            tooltip=[alt.Tooltip("estacion:N", title="Estación"),
                     alt.Tooltip("pm25:Q", title="PM2.5 promedio", format=".1f"),
                     alt.Tooltip("pendiente:Q", title="Tendencia", format=".3f"),
                     alt.Tooltip("estado:N", title="Estado")],
        )
        etiquetas = alt.Chart(datos_chart).mark_text(
            dx=8, dy=-8, fontSize=9, align="left", color=p["TEXTO"]
        ).encode(x="pm25:Q", y="pendiente:Q", text="estacion:N")
        linea_tendencia = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color=p["REFERENCIA"], strokeDash=[5, 3]).encode(y="y:Q")
        linea_eca = alt.Chart(pd.DataFrame({"x": [ECA_PM25_ANUAL]})).mark_rule(
            color=p["REFERENCIA"], strokeDash=[5, 3]).encode(x="x:Q")

        chart = (puntos + etiquetas + linea_tendencia + linea_eca).properties(height=380)

        col_chart, col_texto = st.columns([2, 1])
        with col_chart:
            st.altair_chart(theme.aplicar_estilo_altair(chart), theme=None, width="stretch")
        with col_texto:
            st.caption(
                "Cuadrante superior derecho: estaciones sobre el ECA anual "
                f"({ECA_PM25_ANUAL} µg/m³) y con pendiente positiva — contaminadas y sin "
                "mejora natural."
            )
            st.caption(
                "El eje X es solo PM2.5; el cluster de alta contaminación se calcula con "
                "los 6 contaminantes estandarizados, así que un punto puede quedar cerca "
                "del límite y aun así pertenecer al otro cluster."
            )

    with st.container(border=True):
        st.subheader(":material/priority_high: Distritos a priorizar")
        prioritarias = tabla.loc[tabla["prioridad"], "estacion"].tolist()
        if prioritarias:
            st.success(
                f"Empezar la intervención por: **{', '.join(prioritarias)}** — están en el "
                "cluster de alta contaminación y su tendencia mensual de PM2.5 no muestra "
                "mejora confirmada.",
                icon=":material/priority_high:",
            )
        else:
            st.info(
                "Ninguna estación cumple ambos criterios a la vez con los datos actuales.",
                icon=":material/info:",
            )


if __name__ == "__main__":
    st.set_page_config(page_title="Panel 5 · Recomendaciones", layout="wide")
    render()
