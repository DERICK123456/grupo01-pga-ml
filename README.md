# Predicción de la Aceleración Máxima del Suelo (PGA) con Machine Learning

Trabajo final del curso **Aplicaciones de IA en Estructuras**
Escuela de Postgrado — Universidad Peruana de Ciencias Aplicadas
Docente: Ing. Kurt Soncco Sinchi

## Grupo 01

- Derick De La Vega Sernades
- Diego Kevin Marin Daza
- Jorge Raul Zelaya Mateo

## Descripción

Modelos de Machine Learning (Random Forest, XGBoost y red neuronal) para predecir la aceleración máxima del suelo (PGA) a partir de la magnitud de momento, la distancia a la ruptura, el Vs30, el mecanismo de falla y la profundidad hipocentral, usando la base de datos NGA-West2 del PEER.

Los modelos se comparan contra una línea base tipo GMPE y se auditan con SHAP y curvas de atenuación, bajo el marco Veridical Data Science (PCS) de Yu y Barter (2024).

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `data/` | Instrucciones de descarga del flatfile (los datos no se versionan) |
| `notebooks/` | Análisis exploratorio de datos (T2) |
| `src/` | Scripts de Python (T3) |
| `docs/` | Informes T1, T2 y T3, y bitácora de IA |
| `results/` | Figuras y métricas |

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

El flatfile de NGA-West2 se descarga desde https://ngawest2.berkeley.edu/ (requiere registro gratuito) y se coloca en `data/`. No se incluye en el repositorio por su tamaño y por las condiciones de uso del PEER.

Referencia del dataset: Ancheta, T. D., et al. (2014). NGA-West2 database. *Earthquake Spectra, 30*(3), 989-1005.

## Entregas

| Entrega | Contenido | Peso |
|---|---|---|
| T1 | Definición del problema, dataset, marco PCS y repositorio | 30 % |
| T2 | Análisis exploratorio y plan de algoritmos | 30 % |
| T3 | Implementación, resultados e informe final en formato IEEE | 40 % |

## Uso de IA

El desarrollo se apoyó en asistentes de IA generativa. La bitácora de prompts, iteraciones y validaciones está documentada en `docs/bitacora_ia.md`, conforme a la política de uso de IA del curso.
