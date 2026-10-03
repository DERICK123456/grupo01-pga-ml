"""
Métricas y verificaciones físicas (T2, sección 6, Tabla 7).

Un modelo se considera superior a la línea base solo si cumple dos condiciones
a la vez: menor RMSE en los sismos de prueba Y curvas de atenuación
físicamente coherentes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


# ---------------------------------------------------------------------------
# Métricas de precisión
# ---------------------------------------------------------------------------
def metricas(y_real, y_pred):
    """RMSE, MAE y R² sobre ln(PGA)."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAE": mean_absolute_error(y_real, y_pred),
        "R2": r2_score(y_real, y_pred),
    }


def region_critica(df, mw_min=6.5, r_max=20.0):
    """Máscara de la región que controla el diseño: sismos grandes y cercanos."""
    return (df["Mw"] >= mw_min) & (df[config.DISTANCIA] <= r_max)


def evaluar(modelos, train, test):
    """
    Entrena cada modelo con `train` y lo evalúa en `test`.

    Devuelve una tabla con las métricas globales y las de la región crítica,
    y un diccionario con los modelos ya entrenados.
    """
    X_tr, y_tr = train[config.VARIABLES_ML], train[config.OBJETIVO]
    X_te, y_te = test[config.VARIABLES_ML], test[config.OBJETIVO]
    critica = region_critica(test)

    filas, entrenados = [], {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_te)

        fila = {"modelo": nombre, **metricas(y_te, pred)}
        if critica.sum() > 0:
            fila["RMSE región crítica"] = np.sqrt(
                mean_squared_error(y_te[critica], pred[critica]))
        filas.append(fila)
        entrenados[nombre] = modelo

    tabla = pd.DataFrame(filas).set_index("modelo")
    return tabla, entrenados


# ---------------------------------------------------------------------------
# Verificaciones físicas
# ---------------------------------------------------------------------------
def escenario(Mw, R, Vs30=760.0, prof_hipo=10.0, mecanismo=0):
    """
    Construye entradas sintéticas para trazar curvas de atenuación.

    Por defecto: Vs30 = 760 m/s (referencia de roca en NGA-West2),
    profundidad 10 km y mecanismo 0. Confirmar en la documentación del PEER
    a qué tipo de falla corresponde el código 0 antes de reportarlo.
    """
    R = np.atleast_1d(R).astype(float)
    return pd.DataFrame({
        "Mw": np.full_like(R, Mw),
        "lnR": np.log(np.sqrt(R ** 2 + config.H_SATURACION ** 2)),
        "lnVs30": np.log(Vs30),
        "prof_hipo": prof_hipo,
        "mecanismo": mecanismo,
    })


def curvas_atenuacion(modelo, magnitudes=(5.0, 6.0, 7.0),
                      distancias=None, **kwargs):
    """PGA predicha (en g) en función de la distancia, para varias magnitudes."""
    if distancias is None:
        distancias = np.logspace(0, np.log10(config.R_MAX), 60)
    curvas = {}
    for mw in magnitudes:
        X = escenario(mw, distancias, **kwargs)
        curvas[mw] = np.exp(modelo.predict(X[config.VARIABLES_ML]))
    return pd.DataFrame(curvas, index=pd.Index(distancias, name="R (km)"))


def verificar_monotonia(modelo, tolerancia=1e-6, **kwargs):
    """
    Comprueba la coherencia física de un modelo entrenado:
      - para cada magnitud, la PGA no debe aumentar con la distancia;
      - para cada distancia, la PGA no debe disminuir con la magnitud.

    Devuelve el porcentaje de pasos que violan cada regla (0 % es lo ideal).
    """
    curvas = curvas_atenuacion(modelo, magnitudes=np.arange(4.0, 8.01, 0.25), **kwargs)
    ln = np.log(curvas.values)

    sube_con_distancia = np.diff(ln, axis=0) > tolerancia
    baja_con_magnitud = np.diff(ln, axis=1) < -tolerancia

    return {
        "% violaciones en distancia": 100 * sube_con_distancia.mean(),
        "% violaciones en magnitud": 100 * baja_con_magnitud.mean(),
    }


def residuos(modelo, test):
    """Residuos (observado - predicho) con las variables para graficarlos."""
    pred = modelo.predict(test[config.VARIABLES_ML])
    out = test[["EQID", "Mw", config.DISTANCIA, "Vs30"]].copy()
    out["residuo"] = test[config.OBJETIVO].values - pred
    return out
