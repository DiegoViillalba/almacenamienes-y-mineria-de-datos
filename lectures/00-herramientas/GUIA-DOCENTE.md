# Guía docente · Quarto de cero a GitHub Pages

## Propósito

En 60 minutos, cada estudiante construye y publica una evidencia profesional
breve: un sitio Quarto con narrativa, una celda Python ejecutable, un resultado
interpretado y una URL pública. La sesión prioriza un flujo completo y
repetible sobre un inventario exhaustivo de opciones.

## Resultados de aprendizaje

Al finalizar, el alumnado podrá:

1. distinguir las funciones de Quarto, Python/Jupyter, el editor y Git;
2. explicar la diferencia entre el YAML de un documento y `_quarto.yml`;
3. crear y previsualizar un sitio Quarto de dos páginas;
4. incorporar y controlar una celda Python en un archivo `.qmd`;
5. diferenciar tipo de proyecto y formato de salida;
6. publicar un proyecto en GitHub Pages.

## Preparación antes de la clase

Enviar con al menos 24 horas de anticipación:

- enlaces oficiales para instalar Quarto, Python 3, VS Code y Git;
- solicitud de crear o verificar una cuenta de GitHub;
- petición de instalar las extensiones **Quarto** y **Python** de VS Code;
- los cuatro comandos de verificación: `quarto --version`,
  `python --version`, `python -m jupyter --version` y `git --version`.

En el equipo del docente:

- clonar el repositorio y renderizar la presentación;
- crear un repositorio de GitHub desechable para la demostración;
- tener abierta una versión publicada del proyecto por si falla la red;
- tener TinyTeX instalado, pero no exigir su instalación durante el reloj;
- abrir VS Code, terminal, navegador y notas del presentador antes de iniciar.

## Minuto a minuto

| Minutos | Bloque | Acción del docente | Evidencia observable |
|---:|---|---|---|
| 0–5 | Encuadre | Mostrar el producto final y el cronograma. Formar parejas de apoyo. | Cada persona sabe qué URL debe producir. |
| 5–13 | Entorno | Explicar las cuatro piezas, crear `.venv` y ejecutar verificaciones. | Terminal muestra Quarto, Python, Jupyter y Git. |
| 13–22 | Primer `.qmd` | Escribir `index.qmd`, señalar YAML/Markdown y ejecutar `quarto preview`. | El navegador se actualiza al guardar. |
| 22–32 | Sitio | Crear proyecto website, editar `_quarto.yml` y añadir `analisis.qmd`. | Navegación Inicio/Análisis funciona. |
| 32–42 | Python | Añadir una celda, gráfica, opciones `#|` y una interpretación. | La página muestra resultado + conclusión. |
| 42–50 | Formatos | Separar tipo de proyecto y formato. Mostrar HTML/PDF/DOCX y perfil book. | El alumnado puede decidir qué salida usar. |
| 50–58 | Pages | Inicializar Git, subir a GitHub y ejecutar `quarto publish gh-pages`. | URL pública o despliegue en progreso. |
| 58–60 | Cierre | Recapitular cinco ideas y recoger ticket de salida. | URL o comando bloqueante registrado. |

## Decisiones didácticas

### Construcción incremental

Nunca se inicia un segundo proyecto. `index.qmd` se convierte en portada, se
añade `analisis.qmd`, después código Python y finalmente publicación. El
alumnado percibe que Quarto amplía el mismo artefacto en lugar de sustituirlo.

### Checkpoints, no preguntas genéricas

Evitar “¿todo bien?”. Pedir evidencia concreta:

- “muéstrame las cuatro versiones en la terminal”;
- “cambia una palabra y enséñame el navegador actualizado”;
- “abre Análisis desde la barra”;
- “señala la frase que interpreta tu gráfica”;
- “abre la URL en una ventana privada”.

### Semáforo y trabajo en parejas

Verde ayuda a amarillo; rojo copia `proyecto-demo/` y continúa. El objetivo es
mantener un flujo común, no depurar en público cada particularidad del sistema.

## Contingencias

### Quarto no aparece en PATH

Reiniciar VS Code y la terminal. Si persiste, la persona trabaja con quien está
en verde y completa la instalación al final. No intentar modificar PATH durante
la explicación central.

### PowerShell bloquea la activación

Probar temporalmente en esa terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Si la política institucional lo impide, usar `py -m pip ...` y un compañero con
entorno funcional. No solicitar cambios permanentes de política.

### Jupyter o una biblioteca no están disponibles

Confirmar que el prompt muestra `(.venv)` y ejecutar:

```bash
python -m pip install -r requirements.txt
```

Si la red falla, trabajar con los resultados ya renderizados y conservar el
código para ejecutarlo después.

### TinyTeX tarda o falla

No bloquear la sesión. Mostrar el comando y una salida PDF preparada. HTML es
la ruta principal del taller; PDF es una segunda representación.

### GitHub solicita autenticación

Permitir GitHub Desktop si ya está configurado. De otro modo, realizar el
commit local, observar el despliegue del docente y anotar la publicación como
paso posterior. Nunca crear ni proyectar tokens de acceso durante la clase.

### GitHub Pages devuelve 404

Revisar que el repositorio exista, que `gh-pages` se haya enviado y esperar uno
o dos minutos. Para sitios de usuario, revisar **Settings → Pages** y elegir la
rama `gh-pages` si GitHub no lo hizo automáticamente.

## Rúbrica rápida del producto

Cada criterio vale 0 o 1 punto:

| Criterio | Evidencia |
|---|---|
| Propósito | La portada formula problema y audiencia. |
| Reproducibilidad | Existe al menos una celda Python ejecutable. |
| Comunicación | Un resultado tiene una interpretación escrita. |
| Navegación | Las páginas están conectadas y no hay enlaces rotos. |
| Publicación | La URL abre sin credenciales. |

Con 4/5 el producto ya es compartible; el quinto punto define la mejora
inmediata.

## Después de la sesión

El siguiente ejercicio recomendado es reemplazar los datos ficticios por una
fuente real del proyecto del curso y añadir: descripción de la fuente,
supuestos, una figura con texto alternativo, conclusión y datos de contacto.
