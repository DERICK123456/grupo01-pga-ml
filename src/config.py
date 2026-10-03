"""
Configuración central del proyecto.

Todas las rutas, semillas y decisiones de criterio (judgment calls) se definen
aquí, en un solo lugar. Para evaluar un escenario alternativo en el análisis de
estabilidad basta con cambiar estos valores, sin tocar el resto del código.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
# La raíz se calcula a partir de este archivo, así las rutas funcionan igual
# desde un notebook, un script o la terminal.
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "data"
DIR_RESULTADOS = RAIZ / "results"

FLATFILE = DIR_DATOS / "Updated_NGA_West2_Flatfile_RotD50_d005_public_version.xlsx"

# ---------------------------------------------------------------------------
# Reproducibilidad (principio de Computabilidad, marco PCS)
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Columnas del flatfile -> nombres de trabajo
# ---------------------------------------------------------------------------
COLUMNAS = {
    "Record Sequence Number": "RSN",
    "EQID": "EQID",
    "Earthquake Magnitude": "Mw",
    "Mechanism Based on Rake Angle": "mecanismo",
    "Hypocenter Depth (km)": "prof_hipo",
    "ClstD (km)": "Rrup",
    "Joyner-Boore Dist. (km)": "Rjb",
    "Vs30 (m/s) selected for analysis": "Vs30",
    "Measured/Inferred Class": "vs30_clase",
    "Spectra Quality Flag": "calidad",
    "PGA (g)": "PGA",
}

# Código que el flatfile usa para indicar dato faltante
CODIGO_FALTANTE = -999

# ---------------------------------------------------------------------------
# Decisiones de criterio adoptadas en el T2 (notebook 02, sección 10)
# ---------------------------------------------------------------------------
MW_MIN = 4.0          # Decisión 1: magnitud mínima
R_MAX = 200.0         # Decisión 1: distancia máxima (km)
DISTANCIA = "Rrup"    # Decisión 2: medida de distancia ("Rrup" o "Rjb")
H_SATURACION = 6.0    # km, término de saturación de la forma GMPE

# ---------------------------------------------------------------------------
# Partición
# ---------------------------------------------------------------------------
FRACCION_PRUEBA = 0.2   # 20 % de los sismos se reservan para la prueba final
N_FOLDS = 5             # pliegues de GroupKFold para ajustar hiperparámetros

# ---------------------------------------------------------------------------
# Variables de entrada
# ---------------------------------------------------------------------------
# Forma GMPE de la línea base (definida en el T1 y el T2)
VARIABLES_BASE = ["Mw", "lnR", "lnVs30"]

# Modelos de Machine Learning: mismas variables físicas más profundidad y
# mecanismo de falla
VARIABLES_NUMERICAS = ["Mw", "lnR", "lnVs30", "prof_hipo"]
VARIABLES_CATEGORICAS = ["mecanismo"]
VARIABLES_ML = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS

OBJETIVO = "lnPGA"
