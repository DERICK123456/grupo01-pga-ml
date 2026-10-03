"""
Carga, limpieza y partición del flatfile NGA-West2.

Reúne en funciones reutilizables los pasos validados en los notebooks 01 y 02
del T2, para que todos los notebooks y scripts del T3 usen exactamente el mismo
procesamiento.

Uso desde la terminal (en la raíz del repositorio):
    python -m src.datos
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config


def cargar_flatfile(ruta=config.FLATFILE):
    """Lee solo las columnas necesarias del flatfile y las renombra."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el flatfile en {ruta}.\n"
            "Descárguelo del PEER y colóquelo en la carpeta data/ "
            "(ver README.md)."
        )
    df = pd.read_excel(ruta, usecols=list(config.COLUMNAS))
    return df.rename(columns=config.COLUMNAS)


def limpiar(df):
    """
    Convierte los códigos -999 en nulos y conserva los registros completos.

    El flatfile no deja celdas vacías: codifica los faltantes como -999. Si no
    se convierten, contaminan cualquier estadística (la PGA media sin limpiar
    resulta negativa).
    """
    df = df.copy()
    numericas = df.select_dtypes(include="number").columns
    df[numericas] = df[numericas].replace(config.CODIGO_FALTANTE, np.nan)

    esenciales = ["PGA", "Mw", "Rrup", "Rjb", "Vs30",
                  "mecanismo", "prof_hipo", "EQID"]
    df = df.dropna(subset=esenciales)

    # El EQID y el mecanismo son identificadores o categorías, no magnitudes
    df["EQID"] = df["EQID"].astype(int)
    df["mecanismo"] = df["mecanismo"].astype(int)
    return df


def aplicar_decisiones(df, mw_min=config.MW_MIN, r_max=config.R_MAX,
                       distancia=config.DISTANCIA):
    """
    Aplica las decisiones de criterio del T2 y crea las variables del modelo.

    Los argumentos permiten reproducir los escenarios alternativos del
    análisis de estabilidad sin modificar config.py.
    """
    df = df[(df["Mw"] >= mw_min) & (df[distancia] <= r_max)].copy()

    # Variables transformadas según la forma GMPE (secciones 5.3 del T1 y 4.1 del T2)
    df["lnR"] = np.log(np.sqrt(df[distancia] ** 2 + config.H_SATURACION ** 2))
    df["lnVs30"] = np.log(df["Vs30"])
    df["lnPGA"] = np.log(df["PGA"])
    return df.reset_index(drop=True)


def particion_por_sismo(df, fraccion_prueba=config.FRACCION_PRUEBA,
                        semilla=config.SEED):
    """
    Separa entrenamiento y prueba por sismo (EQID), no por registro.

    Los registros de un mismo sismo comparten la fuente sísmica. Si se
    repartieran al azar, el modelo vería en entrenamiento el mismo evento que
    debe predecir en prueba y la métrica sobreestimaría su capacidad real.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=fraccion_prueba,
                            random_state=semilla)
    idx_tr, idx_te = next(gss.split(df, groups=df["EQID"]))
    train, test = df.iloc[idx_tr], df.iloc[idx_te]

    compartidos = set(train["EQID"]) & set(test["EQID"])
    assert not compartidos, f"Fuga de información: {len(compartidos)} sismos compartidos"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def preparar_datos(**kwargs):
    """Ejecuta todo el procesamiento y devuelve (train, test)."""
    df = aplicar_decisiones(limpiar(cargar_flatfile()), **kwargs)
    return particion_por_sismo(df)


if __name__ == "__main__":
    bruto = cargar_flatfile()
    limpio = limpiar(bruto)
    final = aplicar_decisiones(limpio)
    train, test = particion_por_sismo(final)

    print(f"Flatfile           : {len(bruto):,} filas")
    print(f"Tras limpieza      : {len(limpio):,} registros, {limpio['EQID'].nunique()} sismos")
    print(f"Dataset final      : {len(final):,} registros, {final['EQID'].nunique()} sismos")
    print(f"Entrenamiento      : {len(train):,} registros, {train['EQID'].nunique()} sismos")
    print(f"Prueba             : {len(test):,} registros, {test['EQID'].nunique()} sismos")
