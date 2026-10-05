# Videos cortos de Manim para redes sociales

Guía para producir clips breves (30–60 s) que resumen una clase del curso y se
publican en LinkedIn u otras redes. Complementa a
[`MANIM-SLIDES.md`](MANIM-SLIDES.md): el entorno (`.venv-manim`), la paleta y la
identidad UNAM son los mismos, pero el producto es un **video continuo** (`Scene`,
sin `next_slide`), no una presentación navegable.

Implementación de referencia:
[`03-eda/03-06-medidas-concentracion-manim/linkedin/`](03-eda/03-06-medidas-concentracion-manim/linkedin/)
(CRₖ → HHI → Lorenz → Gini, 44.6 s).

---

## 1. Formato técnico

| Parámetro | Valor | Motivo |
|---|---|---|
| Relación de aspecto | **4:5 vertical** | Ocupa más alto del feed móvil que 16:9 o 1:1 sin recortarse. |
| Resolución | **1080 × 1350** | Máximo nativo de LinkedIn para 4:5. |
| Marco de Manim | `frame_width = 8`, `frame_height = 10` | Coordenadas: x ∈ [−4, 4], y ∈ [−5, 5]. |
| Cuadros por segundo | 30 | Suficiente y ligero (~3.5 MB para 45 s). |
| Duración | 40–50 s | Cabe un recorrido completo sin perder retención. |
| Audio | Ninguno | El feed reproduce en silencio: **todo el mensaje va en pantalla**. |

`manim.cfg` de la carpeta del clip:

```ini
[CLI]
background_color = #07111F
renderer = cairo
media_dir = media
pixel_width = 1080
pixel_height = 1350
frame_width = 8
frame_height = 10
frame_rate = 30
progress_bar = display
verbosity = WARNING
```

- No usar `-ql/-qm/-qh`: sustituyen la resolución del `.cfg`.
- Para vista rápida usar dimensiones **pares** (`-r 544,680 --fps 15`);
  libx264 falla con impares (p. ej. 540 × 675).
- `verbosity = WARNING` silencia el aviso “File ready at…”; si Manim reutiliza
  la caché, el comando termina casi sin salida. Cambiar a `INFO` para verlo.

## 2. Estructura de carpeta

El clip vive dentro de la clase que resume, en una subcarpeta `linkedin/`
(u otra red):

```text
lectures/<unidad>/<clase>-manim/linkedin/
├── README.md                 # guion, comandos y texto de publicación
├── manim.cfg                 # 4:5, 1080 × 1350
├── <tema>_linkedin.py        # una Scene
└── <tema>-linkedin.mp4       # copia final para subir (no se versiona sin decisión del usuario)
```

El script reutiliza `utils.py` de la clase padre (paleta, `safe_text`,
`LOGO_PATH`) insertando `parents[1]` en `sys.path`. `media/` y `__pycache__/`
quedan ignorados por el `.gitignore` de la clase.

## 3. Maquetación vertical (zonas fijas)

Con el marco 8 × 10, reservar bandas estables para que el ojo sepa dónde mirar:

| Zona | y | Contenido |
|---|---|---|
| Navegación | 4.40 | Pasos del recorrido (`CRₖ · HHI · Lorenz · Gini`); el activo en color y subrayado, los demás en `GRID`. |
| Título | 3.50 | Pregunta del acto, 44 pt negrita, `max_width ≈ 7.4`. |
| Visual principal | −2.65 a 2.35 | Cuadro de lado 5 centrado en (0.25, −0.15). |
| Rótulo | −3.55 | Una frase de ≤ 45 caracteres o una fórmula `MathTex`. |
| Firma | −4.62 | “Diego Villalba · Almacenes y Minería de Datos” + `Logo_FC_Blanco.png`, sobre una línea en y = −4.30. |

Tamaños mínimos legibles en móvil: títulos 44, rótulos 30, etiquetas 20–24.

## 4. Principios narrativos

1. **Gancho en 0–3 s.** Pregunta concreta + dato visible
   (“¿Quién acumula el total?” con las barras 5, 10, 15, 20, 50).
2. **Un solo ejemplo** de principio a fin; los números deben coincidir con el
   capítulo del libro.
3. **Hilo geométrico continuo.** Cada acto transforma los objetos del anterior
   en vez de borrarlos: barras → columna apilada (CRₖ) → franjas de un cuadrado
   de parejas (HHI) → escalera acumulada (Lorenz) → curva que se deforma →
   área (Gini).
4. **Geometría antes que fórmula.** La fórmula aparece solo como rótulo final
   del acto, con el número del ejemplo.
5. **Un acto ≈ 6–10 s.** Animaciones de 0.4–1.6 s y pausas de lectura de
   0.6–2 s tras cada rótulo nuevo.
6. **Mostrar el límite** de cada medida en una frase (“Rápido de leer… pero
   ignora el resto”).
7. **Cierre legible 3 s**: resumen de una línea por concepto, mensaje final,
   curso y autoría. El último cuadro funciona como miniatura del bucle.
8. **Color con significado estable**: unidad dominante en `ORANGE`, resto en
   `BLUE`, cada medida con su color en navegación, títulos y cierre.

## 5. Trampas conocidas

| Síntoma | Solución |
|---|---|
| Cuadro vacío en “CRₖ” | La fuente `Sans` no trae `ₖ`: usar `MarkupText("CR<sub>k</sub>")`. |
| `avcodec_open2("libx264")` falla | Dimensiones impares; usar pares. |
| Fórmula tapa la etiqueta del eje | Retirar etiquetas de ejes antes del rótulo con `MathTex`. |
| Curva animada | `ValueTracker` + `always_redraw`; interpolar entre vectores **ya ordenados** para que la curva siga siendo válida. |

## 6. Verificación

1. Vista rápida (`-r 544,680 --fps 15`) y hoja de contacto de fotogramas:

   ```bash
   ffmpeg -ss <t> -i video.mp4 -frames:v 1 -vf scale=272:340 f_<n>.png
   ffmpeg -pattern_type glob -i 'f_*.png' -vf tile=7x2 -frames:v 1 hoja.png
   ```

2. Revisar: texto dentro del marco, sin solapes, glifos correctos, números
   iguales al libro, firma sin tapar contenido.
3. Render final y `ffprobe`: 1080 × 1350, 30 fps, h264 `yuv420p`, duración
   dentro del objetivo.
4. Copiar el MP4 final a la carpeta del clip con un nombre legible.

## 7. Texto de publicación (LinkedIn)

Convenciones acordadas con el autor:

- **Empieza con un gancho** en la primera línea (se ve antes de “ver más”).
- **Bilingüe**: primero español precedido por 🇲🇽, después inglés precedido
  por 🇺🇸. Las banderas son los **únicos** emojis.
- Viñetas con el carácter literal `•` y una línea en blanco entre ellas; los
  guiones de Markdown se pierden al pegar en LinkedIn. Entregar el texto dentro
  de un bloque de código para copiarlo sin formato.
- Una viñeta por medida: qué responde + el número del ejemplo.
- Cerrar con una frase de síntesis y una **pregunta** a la audiencia.
- **El enlace a las notas va en el primer comentario**, no en el post
  (los enlaces externos reducen el alcance). El post lo anuncia; el comentario
  lo incluye en ambos idiomas y aclara que las notas están en español.
- Verificar que el enlace apunte al capítulo correcto (`curl -o /dev/null -w
  "%{http_code}"` debe dar 200).
- 4–5 hashtags, alternando español e inglés.

El README del clip de referencia contiene un ejemplo completo.

## 8. Variante Instagram Reels (9:16)

Implementaciones de referencia:
[`03-eda/03-05-medidas-heterogeneidad-manim/instagram/`](03-eda/03-05-medidas-heterogeneidad-manim/instagram/)
y [`03-eda/03-06-medidas-concentracion-manim/instagram/`](03-eda/03-06-medidas-concentracion-manim/instagram/).

| Parámetro | Valor |
|---|---|
| Resolución | **1080 × 1920** (`frame_width = 8`, `frame_height = 14.2222222`: misma escala de 135 px/unidad que el 4:5, así los tamaños de letra coinciden) |
| Zona segura | Contenido en y ∈ [−4.1, 5.6]; Reels tapa ≈220 px arriba y ≈380 px abajo, y la cuadrícula del perfil recorta al centro 3:4 |
| Vista rápida | `-r 540,960 --fps 15` |
| Firma | Solo nombre y usuario de Instagram (pie + cierre); sin curso ni logo |

Diferencias de ritmo frente al clip de LinkedIn:

- **Lectura primero.** Cada paso lleva una explicación de 1–3 líneas
  (33 pt, ≤ 34 caracteres por línea) en una banda fija bajo el visual, y la
  escena espera `0.6 s + 0.25 s × palabras` (mín. 1.6 s) antes de cambiar.
- **La animación acompaña al texto**: el cambio visual se lanza en el mismo
  `play` que la explicación que lo describe, no antes.
- Resaltar con color, dentro de la frase, la palabra que nombra el objeto
  coloreado en pantalla (`<span foreground=…>`).
- Duración resultante: 80–100 s. Es más largo que el 4:5, a cambio de que
  el video se entienda sin pausar.
