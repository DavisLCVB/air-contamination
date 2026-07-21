"""theme.py — paleta Nord compartida por los paneles con gráficos matplotlib.

Los colores ya se repetían sin estar centralizados (panel_eda.py, forecast.py
usaban los mismos hex de memoria); este módulo evita esa duplicación y aplica
el mismo estilo de grid/spines a las figuras de los distintos paneles.
"""
from __future__ import annotations

# Paleta Nord (https://www.nordtheme.com/) — la misma que ya usaban
# panel_eda.py y forecast.py de forma dispersa; aquí queda en un solo lugar.
AZUL = "#5e81ac"          # nord10 — normal / baja contaminación / serie principal
ROJO = "#bf616a"          # nord11 — alerta / alta contaminación
VERDE = "#a3be8c"         # nord14 — positivo / valor mínimo destacado
NARANJA = "#d08770"       # nord12 — advertencia intermedia
GRIS_TEXTO = "#3b4252"    # nord1 — texto de series/históricos
GRIS_CLARO = "#d8dee9"    # nord4 — grid y líneas de referencia tenues

COLORMAP_DIVERGENTE = "RdBu_r"  # correlaciones y escalas −1..1 (rojo=alto, azul=bajo)
COLORMAP_CLUSTERS = "Set2"      # paleta cualitativa para clusters/categorías


def aplicar_estilo_mpl(ax) -> None:
    """Grid horizontal tenue + sin spines superior/derecho, igual en todos los paneles."""
    ax.grid(axis="y", color=GRIS_CLARO, linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
