# Taller Quarto en 60 minutos

Paquete autocontenido para impartir una sesión práctica de introducción a
Quarto. Todo el material específico del taller vive en este directorio. La
presentación solo reutiliza dos activos institucionales compartidos del curso:
`../assets/unam-theme.scss` y `../assets/Logo_FC_Color.png`.

## Contenido

```text
00-herramientas/
├── 00-01-quarto-de-cero-a-pages.qmd  # presentación Reveal.js
├── taller-quarto.css                 # componentes visuales del taller
├── GUIA-DOCENTE.md                   # minuto a minuto y contingencias
├── README.md                         # este archivo
└── proyecto-demo/                    # base funcional para docente y alumnado
    ├── _quarto.yml                   # versión sitio web
    ├── _quarto-book.yml              # perfil alternativo de libro
    ├── index.qmd
    ├── analisis.qmd
    ├── conclusiones.qmd
    ├── styles.css
    ├── requirements.txt
    └── workflow-publish.yml.example
```

## Previsualizar la presentación

Desde la raíz del repositorio:

```bash
quarto preview lectures/00-herramientas/00-01-quarto-de-cero-a-pages.qmd
```

La salida compilada se escribe, según la convención del repositorio, en:

```text
lectures/_output/00-herramientas/00-01-quarto-de-cero-a-pages.html
```

## Probar el proyecto de práctica

```bash
cd lectures/00-herramientas/proyecto-demo
python3 -m venv .venv          # en Windows: py -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
quarto preview
```

Para demostrar que el mismo contenido puede tomar forma de libro:

```bash
quarto render --profile book --to html
```

La salida del sitio se guarda en `_site/`; la del perfil de libro, en
`_book/`. Para PDF se necesita TinyTeX:

```bash
quarto install tinytex
quarto render --profile book --to pdf
```

## Publicación

La ruta didáctica principal usa una primera publicación local y controlada:

```bash
quarto publish gh-pages
```

Después de esa primera publicación, el archivo
`workflow-publish.yml.example` puede copiarse a
`.github/workflows/publish.yml` dentro del repositorio del estudiante para
automatizar los siguientes despliegues.
