<div align="center">

# Almacenes y Minería de Datos
### Data Warehousing & Data Mining

**Facultad de Ciencias · UNAM · Semestre 2026-2**

[![Course Site](https://img.shields.io/badge/Course%20Site-GitHub%20Pages-0a0a0a?style=flat-square)](https://diegoviillalba.github.io/almacenamienes-y-mineria-de-datos/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Quarto](https://img.shields.io/badge/Built%20with-Quarto-75AADB?style=flat-square)](https://quarto.org/)

</div>

---

> [Español](#español) · [English](#english)

---

<a name="español"></a>

## Acerca del repositorio

Este repositorio contiene los materiales del curso **Almacenes y Minería de Datos** de la **Facultad de Ciencias, UNAM**. Los materiales están organizados como un **Quarto Book** desplegado en GitHub Pages, e incluyen:

- **Notas de clase** — libro en formato Quarto con desarrollo completo de cada tema.
- **Slides** — presentaciones de cada sesión en formato Quarto Reveal.js.
- **Material de ejercicios** — notebooks y conjuntos de datos para práctica y evaluación.

El sitio del curso se encuentra en:
**[https://diegoviillalba.github.io/almacenamienes-y-mineria-de-datos/](https://diegoviillalba.github.io/almacenamienes-y-mineria-de-datos/)**

---

## Objetivos del curso

Al finalizar el curso, el estudiante sera capaz de:

- Comprender el rol de un **Data Warehouse** y su diferencia con sistemas operacionales (OLTP).
- Disenar **esquemas dimensionales** (Kimball) y discutir arquitecturas alternativas (Inmon).
- Implementar flujos **ETL/ELT** reproducibles con validacion de calidad e integracion de fuentes.
- Construir y explotar **cubos de datos** multidimensionales mediante operaciones OLAP.
- Aplicar tecnicas de **mineria de datos**: clasificacion, agrupamiento, reglas de asociacion y prediccion.
- Comunicar resultados con rigor: metricas, validacion cruzada, interpretacion y visualizacion.

---

## Contenido tematico

**Unidad I — Almacenes de Datos**

| | |
|---|---|
| I.1 | Introduccion |
| I.2 | Un modelo de datos multidimensional |
| I.3 | Arquitectura de un almacen de datos |
| I.4 | Implementacion de un almacen de datos |
| I.5 | Relacion entre los almacenes de datos y la mineria de datos |

**Unidad II — Pre-procesamiento de Datos**

| | |
|---|---|
| II.1 | Introduccion |
| II.2 | El pre-procesamiento de datos |
| II.3 | Resumen descriptivo de datos |
| II.4 | Limpieza de datos |
| II.5 | Integracion y transformacion de datos |
| II.6 | Reduccion de datos |
| II.7 | Generacion de jerarquias |

**Unidad III — Computacion de Cubos de Datos y Generalizaciones**

| | |
|---|---|
| III.1 | Calculo eficiente de cubos de datos |
| III.2 | Exploracion y descubrimiento de informacion en bases multidimensionales |
| III.3 | Construccion y explotacion de cubos |

**Unidad IV — Minado de Patrones Frecuentes, Asociaciones y Correlaciones**

| | |
|---|---|
| IV.1 | Conceptos basicos |
| IV.2 | Minado eficiente y escalable de conjuntos de elementos frecuentes |
| IV.3 | Minando varios tipos de reglas de asociacion |
| IV.4 | Del minado de reglas de asociacion al analisis de correlaciones |

**Unidad V — Modelos de Clasificacion y Prediccion**

| | |
|---|---|
| V.1 | Aprendizaje supervisado |
| V.2 | Arboles de decision |
| V.3 | Clasificador Bayesiano |
| V.4 | Clasificacion basada en reglas |
| V.5 | Redes neuronales |
| V.6 | Otros metodos de clasificacion |

**Unidad VI — Analisis de Agrupamiento**

| | |
|---|---|
| VI.1 | Introduccion |
| VI.2 | Tipos de datos |
| VI.3 | Categorizacion de metodos de agrupamiento |
| VI.4 | Agrupando datos de alta dimensionalidad |
| VI.5 | Analisis de valores atipicos (Outliers) |
| VI.6 | Metodos Jerarquicos |
| VI.7 | Metodos basados en densidad |
| VI.8 | Modelos basados en metodos de agrupamiento |

---

## Estructura del repositorio

```
notas-mineria/
├── _quarto.yml              # Configuracion del Quarto Book
├── index.qmd                # Pagina principal
├── unidad-1/                # Almacenes de Datos
├── unidad-2/                # Pre-procesamiento
├── unidad-3/                # Cubos de Datos
├── unidad-4/                # Patrones y Asociaciones
├── unidad-5/                # Clasificacion y Prediccion
├── unidad-6/                # Agrupamiento
├── slides/                  # Presentaciones por sesion
└── ejercicios/              # Notebooks y datasets
```

---

---

<a name="english"></a>

## About this Repository

This repository contains the course materials for **Data Warehousing & Data Mining** at the **Faculty of Sciences, UNAM**. Materials are organized as a **Quarto Book** deployed on GitHub Pages, and include:

- **Lecture notes** — a Quarto Book with full coverage of each topic.
- **Slides** — per-session presentations in Quarto Reveal.js format.
- **Exercise materials** — notebooks and datasets for practice and assessment.

The course site is available at:
**[https://diegoviillalba.github.io/almacenamienes-y-mineria-de-datos/](https://diegoviillalba.github.io/almacenamienes-y-mineria-de-datos/)**

---

## Course Objectives

By the end of the course, students will be able to:

- Understand the role of a **Data Warehouse** and how it differs from operational (OLTP) systems.
- Design **dimensional schemas** (Kimball) and discuss alternative architectures (Inmon).
- Implement reproducible **ETL/ELT** pipelines with quality validation and source integration.
- Build and exploit **multidimensional data cubes** using OLAP operations.
- Apply **data mining** techniques: classification, clustering, association rules, and prediction.
- Communicate results rigorously: metrics, cross-validation, interpretation, and visualization.

---

## Course Outline

| Unit | Topic |
|------|-------|
| I | Data Warehouses |
| II | Data Pre-processing |
| III | Data Cube Computation & Generalizations |
| IV | Frequent Pattern Mining, Associations & Correlations |
| V | Classification & Prediction Models |
| VI | Cluster Analysis |

---

<div align="center">

Facultad de Ciencias · UNAM · 2026

</div>

---

## Autor / Author

Estas notas fueron elaboradas por **Diego Villalba** como material de apoyo para el curso.
Para dudas, comentarios o correcciones, puedes escribir a:
[diego.villalba@ciencias.unam.mx](mailto:diego.villalba@ciencias.unam.mx)

---

These notes were written by **Diego Villalba** as course support material.
For questions, comments, or corrections, feel free to reach out at:
[diego.villalba@ciencias.unam.mx](mailto:diego.villalba@ciencias.unam.mx)
