"""
Definición de los modelos a comparar (plan de algoritmos del T2, sección 5.2).

Cada función devuelve un Pipeline de scikit-learn que incluye su propio
preprocesamiento. Así cada modelo recibe exactamente lo que necesita
(escalado, codificación one-hot) y el preprocesamiento se ajusta solo con los
datos de entrenamiento, sin fuga de información hacia la prueba.

Los hiperparámetros definidos aquí son valores iniciales razonables. Su ajuste
se realiza en el paso 2 mediante validación cruzada agrupada por sismo.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src import config


def _preprocesador_escalado():
    """Escala las numéricas y aplica one-hot al mecanismo de falla.

    El mecanismo está codificado como 0, 1, 2, 3, 4. Para modelos lineales y
    redes neuronales esa codificación implicaría un orden entre categorías que
    no existe; el one-hot lo evita.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), config.VARIABLES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.VARIABLES_CATEGORICAS),
    ])


def linea_base():
    """Regresión lineal con la forma GMPE: ln(PGA) = c0 + c1*Mw + c2*lnR + c3*lnVs30."""
    return Pipeline([
        ("seleccion", ColumnTransformer(
            [("gmpe", "passthrough", config.VARIABLES_BASE)])),
        ("modelo", LinearRegression()),
    ])


def random_forest(**params):
    """Random Forest. Los árboles no requieren escalado ni one-hot."""
    defaults = dict(n_estimators=300, min_samples_leaf=5,
                    n_jobs=-1, random_state=config.SEED)
    defaults.update(params)
    return Pipeline([
        ("seleccion", ColumnTransformer(
            [("vars", "passthrough", config.VARIABLES_ML)])),
        ("modelo", RandomForestRegressor(**defaults)),
    ])


def _restricciones_monotonia():
    """
    Restricciones físicas para XGBoost, en el orden de VARIABLES_ML:
      Mw        -> +1 : la PGA solo puede aumentar con la magnitud
      lnR       -> -1 : la PGA solo puede disminuir con la distancia
      lnVs30    ->  0 : sin restricción (la respuesta del sitio es no lineal)
      prof_hipo ->  0 : sin restricción
      mecanismo ->  0 : categórica, sin orden
    """
    signos = {"Mw": 1, "lnR": -1}
    return tuple(signos.get(v, 0) for v in config.VARIABLES_ML)


def xgboost(monotono=False, **params):
    """XGBoost, opcionalmente con restricciones de monotonía física."""
    defaults = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=config.SEED, n_jobs=-1)
    if monotono:
        defaults["monotone_constraints"] = _restricciones_monotonia()
    defaults.update(params)
    return Pipeline([
        ("seleccion", ColumnTransformer(
            [("vars", "passthrough", config.VARIABLES_ML)])),
        ("modelo", XGBRegressor(**defaults)),
    ])


def red_neuronal(**params):
    """Perceptrón multicapa con parada temprana para evitar sobreajuste."""
    defaults = dict(hidden_layer_sizes=(64, 32), alpha=1e-3,
                    learning_rate_init=1e-3, max_iter=500,
                    early_stopping=True, random_state=config.SEED)
    defaults.update(params)
    return Pipeline([
        ("prep", _preprocesador_escalado()),
        ("modelo", MLPRegressor(**defaults)),
    ])


def todos_los_modelos():
    """Diccionario con los cinco modelos a comparar."""
    return {
        "Línea base (GMPE)": linea_base(),
        "Random Forest": random_forest(),
        "XGBoost": xgboost(monotono=False),
        "XGBoost monótono": xgboost(monotono=True),
        "Red neuronal": red_neuronal(),
    }
