"""Generadores de figuras estáticas (matplotlib/SHAP) para el Panel 2, el Panel 3 y
el reporte/notebooks. No son parte del cálculo (`core/`): son la representación
visual de resultados ya calculados.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from application import theme


def figura_matriz_confusion(resultado: dict, ruta_png: Path, titulo: str | None = None) -> None:
    """Guarda la matriz de confusión anotada (TP/TN/FP/FN) como PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = resultado["matriz_confusion"]
    mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    etiquetas = np.array(
        [[f"TN\n{m['tn']}", f"FP\n{m['fp']}"], [f"FN\n{m['fn']}", f"TP\n{m['tp']}"]]
    )

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, etiquetas[i, j], ha="center", va="center",
                color="white" if mat[i, j] > mat.max() / 2 else "black",
                fontsize=12, fontweight="bold",
            )
    ax.set_xticks([0, 1], ["Pred. baja", "Pred. alta"])
    ax.set_yticks([0, 1], ["Real baja", "Real alta"])
    ax.set_title(titulo or resultado.get("nombre", "Matriz de confusión"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _nombres_legibles(nombres_encoding: list[str]) -> list[str]:
    """Quita el prefijo `estacion__`/`remainder__` que deja ColumnTransformer."""
    return [n.split("__", 1)[-1] if "__" in n else n for n in nombres_encoding]


def explicar_shap(
    modelo: Any, X, dir_salida: Path, prefijo: str, n_muestra: int = 500, idx_instancia: int = 0,
) -> dict[str, str]:
    """Genera summary_plot (global) y force_plot (local) con SHAP; devuelve rutas de los PNG.

    `modelo` es un Pipeline (`prep` + `clf`, ver core/models.py): TreeExplainer necesita
    el clasificador crudo, así que se explica sobre la matriz ya codificada por `prep`
    (incl. el one-hot de `estacion`), no sobre las columnas de entrada originales.
    """
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.preprocessing import SEED

    if len(X) > n_muestra:
        idx = np.random.RandomState(SEED).choice(len(X), size=n_muestra, replace=False)
        X_s = X.iloc[idx]
    else:
        X_s = X

    prep = modelo.named_steps["prep"]
    clf = modelo.named_steps["clf"]
    X_np = prep.transform(X_s)
    nombres = _nombres_legibles(list(prep.get_feature_names_out()))

    explainer = shap.TreeExplainer(clf)
    valores = explainer.shap_values(X_np)

    # Normaliza a los valores SHAP de la clase positiva (RF -> lista/3D; XGB -> 2D)
    if isinstance(valores, list):
        sv = valores[1]
    elif getattr(valores, "ndim", 2) == 3:
        sv = valores[:, :, 1]
    else:
        sv = valores
    base = explainer.expected_value
    base = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) else base

    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta_summary = dir_salida / f"{prefijo}_shap_summary.png"
    ruta_force = dir_salida / f"{prefijo}_shap_force.png"

    plt.figure()
    shap.summary_plot(sv, X_np, feature_names=nombres, show=False)
    plt.tight_layout()
    plt.savefig(ruta_summary, dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.force_plot(
        base, sv[idx_instancia], np.round(X_np[idx_instancia], 2),
        feature_names=nombres, matplotlib=True, show=False,
    )
    plt.savefig(ruta_force, dpi=130, bbox_inches="tight")
    plt.close()

    return {"summary": str(ruta_summary), "force": str(ruta_force)}


def graficar(paquete: dict, ruta_png: Path | None = None):
    """Historia + ajuste sobre test + pronóstico futuro (con IC si el modelo lo da)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = theme.paleta()
    serie, test, mejor = paquete["serie"], paquete["test"], paquete["mejor"]
    yhat_test = paquete["resultados"][mejor]["yhat"]
    fut = paquete["futuro"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(serie.index, serie.values, color=p["TEXTO"], lw=1.2, label="Histórico", zorder=2)
    ax.plot(yhat_test.index, yhat_test.values, "--", color=p["ROJO"], lw=1.6,
            label=f"Ajuste test ({mejor})", zorder=2)
    ax.plot(fut.index, fut["yhat"], color=p["AZUL"], lw=2, marker="o", label="Pronóstico", zorder=2)
    if "lo" in fut.columns:
        ax.fill_between(fut.index, fut["lo"], fut["hi"], color=p["AZUL"], alpha=0.18, label="IC 80%")
    ax.set_title(f"PM2.5 — {paquete['config']['estacion']} · mejor: {mejor} · "
                 f"MAPE {paquete['tabla'].loc[mejor,'mape']:.1f}% · "
                 f"RMSE {paquete['tabla'].loc[mejor,'rmse']:.1f}")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper left", fontsize=8)
    theme.aplicar_estilo_mpl(ax)
    fig.tight_layout()
    if ruta_png:
        ruta_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta_png, dpi=120)
    return fig
