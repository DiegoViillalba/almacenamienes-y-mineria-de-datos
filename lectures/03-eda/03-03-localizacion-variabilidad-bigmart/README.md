# Misión BigMart · materiales de clase

Paquete para una clase-taller de 50 minutos sobre medidas de localización y
variabilidad aplicadas a BigMart.

## Archivos

- `03-03-localizacion-variabilidad-bigmart.qmd`: presentación Reveal.js.
- `notebook-alumnos.ipynb`: notebook incompleto para los alumnos.
- `guion-docente.md`: conducción minuto a minuto, respuestas y contingencias.
- `clase-interactiva.css` y `timer.html`: apariencia y temporizador de la presentación.

## Presentación

La presentación usa el tema `default` de Reveal.js, el logotipo de la FC grande
en la portada y pequeño en las diapositivas siguientes. Mantiene el pie vacío y
los márgenes reducidos. Las cuatro gráficas principales usan Plotly en una
diapositiva completa; permiten consultar valores, hacer zoom, desplazar la vista
y restaurarla desde la barra de herramientas. En los ejemplos extensos, avance
con la flecha derecha para destacar el código por bloques lógicos.

Desde la raíz del repositorio:

```bash
quarto render lectures/03-eda/03-03-localizacion-variabilidad-bigmart/03-03-localizacion-variabilidad-bigmart.qmd
```

La salida se genera en:

```text
lectures/_output/03-eda/03-03-localizacion-variabilidad-bigmart/03-03-localizacion-variabilidad-bigmart.html
```

Las diapositivas de actividad tienen un contador automático. Use `Alt+T` para
pausar/continuar, `Alt+R` para reiniciar y `Alt+H` para ocultarlo.

## Notebook local

```bash
.env/bin/jupyter lab lectures/03-eda/03-03-localizacion-variabilidad-bigmart/notebook-alumnos.ipynb
```

El notebook busca `data/bigmart_sales.csv` recorriendo las carpetas superiores.
Si se abre fuera del repositorio, intenta leer el dataset desde GitHub.

## Abrir en Colab

[Abrir notebook de alumnos en Google Colab](https://colab.research.google.com/github/DiegoViillalba/almacenamienes-y-mineria-de-datos/blob/main/lectures/03-eda/03-03-localizacion-variabilidad-bigmart/notebook-alumnos.ipynb)

[Descargar directamente `notebook-alumnos.ipynb`](https://raw.githubusercontent.com/DiegoViillalba/almacenamienes-y-mineria-de-datos/main/lectures/03-eda/03-03-localizacion-variabilidad-bigmart/notebook-alumnos.ipynb)

El enlace funcionará una vez que los archivos estén publicados en la rama
`main`. Para una clase sin red, clone el repositorio con anticipación.

## Dependencias

- Python 3
- pandas
- NumPy
- Plotly
- Jupyter

No se requieren widgets, servicios externos ni instalaciones durante la clase.
