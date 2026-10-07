"""
Ajuste de hiperparámetros con validación cruzada agrupada por sismo.

Para cada modelo se exploran combinaciones al azar dentro de un espacio de
búsqueda (RandomizedSearchCV) y cada combinación se evalúa con GroupKFold por
EQID: en cada pliegue, los sismos de validación nunca aparecen en el
entrenamiento. El conjunto de prueba (81 sismos) NO se usa en ninguna parte de
este proceso; se reserva para la evaluación final.

Los mejores hiperparámetros se guardan en results/mejores_hiperparametros.json
para que la evaluación final use exactamente los mismos valores.

Uso desde la terminal (en la raíz del repositorio):
    python -m src.ajuste
"""

import json
import time

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from src import config, modelos

ARCHIVO_PARAMETROS = config.DIR_RESULTADOS / "mejores_hiperparametros.json"

# ---------------------------------------------------------------------------
# Espacios de búsqueda
# ---------------------------------------------------------------------------
# Los prefijos "modelo__" apuntan al paso "modelo" de cada Pipeline.
# Cada rango se eligió para cubrir desde modelos simples (poco riesgo de
# sobreajuste) hasta modelos flexibles, de modo que la validación cruzada
# decida el equilibrio.
_ESPACIO_XGB = {
    "modelo__n_estimators": randint(200, 1201),      # número de árboles
    "modelo__learning_rate": loguniform(0.01, 0.2),  # tamaño de cada paso
    "modelo__max_depth": randint(2, 9),              # complejidad de cada árbol
    "modelo__min_child_weight": [1, 3, 5, 10],       # mínimo de datos por hoja
    "modelo__subsample": uniform(0.6, 0.4),          # fracción de registros (0.6-1.0)
    "modelo__colsample_bytree": uniform(0.6, 0.4),   # fracción de variables (0.6-1.0)
    "modelo__reg_lambda": loguniform(0.1, 10),       # regularización L2
}

ESPACIOS = {
    "Random Forest": {
        "modelo__n_estimators": [200, 400, 600],
        "modelo__max_depth": [None, 10, 20, 30],
        "modelo__min_samples_leaf": [1, 2, 5, 10, 20],
        "modelo__max_features": [1.0, 0.8, 0.6, "sqrt"],
    },
    "XGBoost": _ESPACIO_XGB,
    "XGBoost monótono": _ESPACIO_XGB,
    "Red neuronal": {
        "modelo__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
        "modelo__alpha": loguniform(1e-5, 1e-1),          # regularización L2
        "modelo__learning_rate_init": loguniform(1e-4, 1e-2),
    },
}

CONSTRUCTORES = {
    "Random Forest": lambda: modelos.random_forest(),
    "XGBoost": lambda: modelos.xgboost(monotono=False),
    "XGBoost monótono": lambda: modelos.xgboost(monotono=True),
    "Red neuronal": lambda: modelos.red_neuronal(),
}


def ajustar(nombre, train, n_iter=20, semilla=config.SEED):
    """
    Busca los mejores hiperparámetros de un modelo.

    Devuelve el objeto RandomizedSearchCV ya ajustado: contiene la mejor
    combinación (best_params_) y el registro completo de todas las
    combinaciones probadas (cv_results_), necesario para justificar la
    elección en el informe.
    """
    busqueda = RandomizedSearchCV(
        estimator=CONSTRUCTORES[nombre](),
        param_distributions=ESPACIOS[nombre],
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=config.N_FOLDS),
        random_state=semilla,
        refit=False,          # el modelo final se entrena en la evaluación
        n_jobs=1,             # cada modelo ya usa todos los núcleos
        return_train_score=True,
    )
    busqueda.fit(train[config.VARIABLES_ML], train[config.OBJETIVO],
                 groups=train["EQID"])
    return busqueda


def resumen_busqueda(busqueda, top=5):
    """Tabla con las mejores combinaciones: RMSE de validación y de entrenamiento."""
    r = pd.DataFrame(busqueda.cv_results_)
    tabla = pd.DataFrame({
        "RMSE validación": -r["mean_test_score"],
        "desv. validación": r["std_test_score"],
        "RMSE entrenamiento": -r["mean_train_score"],
    })
    # La brecha entre entrenamiento y validación indica sobreajuste
    tabla["brecha"] = tabla["RMSE validación"] - tabla["RMSE entrenamiento"]
    params = r["params"].apply(
        lambda p: {k.replace("modelo__", ""): v for k, v in p.items()})
    tabla["hiperparámetros"] = params
    return tabla.sort_values("RMSE validación").head(top)


def _a_json(valor):
    """Convierte tipos de numpy y tuplas a tipos que JSON acepta."""
    if hasattr(valor, "item"):
        return valor.item()
    if isinstance(valor, tuple):
        return list(valor)
    return valor


def guardar(mejores):
    """Guarda {modelo: {parámetro: valor}} en results/."""
    config.DIR_RESULTADOS.mkdir(exist_ok=True)
    limpio = {m: {k.replace("modelo__", ""): _a_json(v) for k, v in p.items()}
              for m, p in mejores.items()}
    ARCHIVO_PARAMETROS.write_text(json.dumps(limpio, indent=2, ensure_ascii=False),
                                  encoding="utf-8")


def cargar():
    """Lee los mejores hiperparámetros guardados por `guardar`."""
    datos = json.loads(ARCHIVO_PARAMETROS.read_text(encoding="utf-8"))
    # JSON no tiene tuplas: se restauran para las capas de la red neuronal
    if "Red neuronal" in datos and "hidden_layer_sizes" in datos["Red neuronal"]:
        datos["Red neuronal"]["hidden_layer_sizes"] = tuple(
            datos["Red neuronal"]["hidden_layer_sizes"])
    return datos


def modelos_ajustados():
    """Los cinco modelos, construidos con los hiperparámetros guardados."""
    p = cargar()
    return {
        "Línea base (GMPE)": modelos.linea_base(),
        "Random Forest": modelos.random_forest(**p["Random Forest"]),
        "XGBoost": modelos.xgboost(monotono=False, **p["XGBoost"]),
        "XGBoost monótono": modelos.xgboost(monotono=True, **p["XGBoost monótono"]),
        "Red neuronal": modelos.red_neuronal(**p["Red neuronal"]),
    }


if __name__ == "__main__":
    from src.datos import preparar_datos

    train, _ = preparar_datos()   # la prueba no se usa en el ajuste
    mejores = {}
    for nombre in CONSTRUCTORES:
        inicio = time.time()
        busqueda = ajustar(nombre, train)
        mejores[nombre] = busqueda.best_params_
        print(f"{nombre:18s} RMSE validación: {-busqueda.best_score_:.3f}"
              f"   ({time.time() - inicio:.0f} s)")
    guardar(mejores)
    print(f"\nGuardado en {ARCHIVO_PARAMETROS}")
