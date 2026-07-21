"""theme.py — paleta Catppuccin (Mocha oscuro / Latte claro) compartida por los paneles.

El modo activo se guarda en `st.session_state["modo_claro"]` (toggle en el
sidebar de app.py). `paleta()` devuelve el diccionario de colores vigente y
`aplicar_estilo_mpl` lo usa para que los gráficos matplotlib seudo respondan
al mismo cambio que la UI nativa de Streamlit (ver `inyectar_css`).

`.streamlit/config.toml` define el tema oscuro (Mocha) como *default* del
servidor — Streamlit no permite cambiar esa config en caliente desde Python,
así que el modo claro se logra inyectando un `<style>` que sobreescribe los
colores del chrome nativo (fondo, sidebar, texto). Es un mejor-esfuerzo: cubre
los contenedores principales: los widgets muy internos de Streamlit podrían
no recolorear al 100%; el selector de tema nativo (menú ☰ → Settings) sigue
disponible como alternativa 100% soportada si algo no se ve bien.
"""
from __future__ import annotations

import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Catppuccin Mocha (oscuro) — https://catppuccin.com/palette
_MOCHA = {
    "MAUVE": "#cba6f7",
    "PINK": "#f5c2e7",
    "ROJO": "#f38ba8",           # red — alerta / alta contaminación
    "NARANJA": "#fab387",        # peach — advertencia intermedia
    "AMARILLO": "#f9e2af",
    "VERDE": "#a6e3a1",          # green — positivo / valor mínimo destacado
    "TEAL": "#94e2d5",
    "AZUL": "#89b4fa",           # blue — normal / baja contaminación / serie principal
    "LAVANDA": "#b4befe",
    "TEXTO": "#cdd6f4",
    "SUPERFICIE": "#313244",     # surface0 — fondo de ejes
    "SUPERFICIE_2": "#585b70",   # surface2 — grid / punto medio divergente
    "FONDO": "#1e1e2e",          # base
    "SIDEBAR": "#181825",        # mantle
    "REFERENCIA": "#9399b2",     # overlay2
    "REFERENCIA_TENUE": "#6c7086",  # overlay0
}

# Catppuccin Latte (claro) — misma paleta, variante clara oficial.
_LATTE = {
    "MAUVE": "#8839ef",
    "PINK": "#ea76cb",
    "ROJO": "#d20f39",
    "NARANJA": "#fe640b",
    "AMARILLO": "#df8e1d",
    "VERDE": "#40a02b",
    "TEAL": "#179299",
    "AZUL": "#1e66f5",
    "LAVANDA": "#7287fd",
    "TEXTO": "#4c4f69",
    "SUPERFICIE": "#e6e9ef",
    "SUPERFICIE_2": "#ccd0da",
    "FONDO": "#eff1f5",
    "SIDEBAR": "#e6e9ef",
    "REFERENCIA": "#8c8fa1",
    "REFERENCIA_TENUE": "#9ca0b0",
}


def modo_claro() -> bool:
    """True si el usuario activó el toggle de modo claro en el sidebar."""
    return bool(st.session_state.get("modo_claro", False))


def paleta() -> dict[str, str]:
    """Diccionario de colores vigente según el modo (claro/oscuro) activo."""
    return _LATTE if modo_claro() else _MOCHA


def colormap_divergente():
    """Escala azul (bajo) -> gris (medio) -> rojo (alto), ver panel_eda.py."""
    p = paleta()
    return LinearSegmentedColormap.from_list(
        "catppuccin_diverging", [p["AZUL"], p["SUPERFICIE_2"], p["ROJO"]]
    )


def colormap_clusters():
    p = paleta()
    return ListedColormap(
        [p["MAUVE"], p["VERDE"], p["NARANJA"], p["AZUL"], p["PINK"], p["TEAL"],
         p["AMARILLO"], p["LAVANDA"]],
        name="catppuccin_clusters",
    )


def aplicar_estilo_mpl(ax) -> None:
    """Aplica el tema (fondo, texto, grid, spines, leyenda) vigente a una figura matplotlib."""
    p = paleta()
    fig = ax.figure
    fig.patch.set_facecolor(p["FONDO"])
    ax.set_facecolor(p["SUPERFICIE"])

    ax.grid(axis="y", color=p["SUPERFICIE_2"], linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("bottom", "left"):
        ax.spines[lado].set_color(p["SUPERFICIE_2"])

    ax.tick_params(colors=p["TEXTO"], labelcolor=p["TEXTO"])
    ax.xaxis.label.set_color(p["TEXTO"])
    ax.yaxis.label.set_color(p["TEXTO"])
    if ax.get_title():
        ax.title.set_color(p["TEXTO"])

    leyenda = ax.get_legend()
    if leyenda is not None:
        leyenda.get_frame().set_facecolor(p["SUPERFICIE"])
        leyenda.get_frame().set_edgecolor(p["SUPERFICIE_2"])
        for texto in leyenda.get_texts():
            texto.set_color(p["TEXTO"])


def inyectar_css() -> None:
    """Sobreescribe el chrome nativo de Streamlit con la paleta clara, si está activa.

    `.streamlit/config.toml` fija Mocha (oscuro) como tema de arranque del
    servidor; Streamlit no expone una API para cambiarlo en caliente. Esto
    inyecta CSS que repinta los contenedores principales (fondo, sidebar,
    texto, botones) para simular el modo claro sin reiniciar el servidor.
    """
    if not modo_claro():
        return
    p = paleta()
    st.markdown(
        f"""
        <style>
        :root {{
            --primary-color: {p['MAUVE']};
            --background-color: {p['FONDO']};
            --secondary-background-color: {p['SUPERFICIE']};
            --text-color: {p['TEXTO']};
        }}
        [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {{
            background-color: {p['FONDO']};
            color: {p['TEXTO']};
        }}
        [data-testid="stSidebar"] {{
            background-color: {p['SIDEBAR']};
            color: {p['TEXTO']};
        }}
        [data-testid="stHeader"] {{
            background-color: transparent;
        }}
        [data-testid="stMarkdownContainer"], p, span, label, li {{
            color: {p['TEXTO']};
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: {p['TEXTO']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
