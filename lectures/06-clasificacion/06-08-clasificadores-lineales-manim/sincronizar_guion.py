"""Genera el guion desde la misma narración que las notas y el manifiesto final."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parent
entries = json.loads((ROOT / 'narracion.json').read_text())
manifest = json.loads((ROOT / 'slides/ClasificadoresLineales.json').read_text())
assert len(entries) == len(manifest['slides']) == 65, 'No usar un render parcial.'

def timestamp(seconds):
    seconds = round(seconds)
    return f'{seconds // 60:02}:{seconds % 60:02}'

def duration(file):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration','-of','default=nw=1:nk=1',str(file)],text=True))

words = sum(len(e['narration'].split()) for e in entries.values())
text = f'''# Guion de grabación · Clasificadores lineales

Edición española para YouTube · Diego Villalba · Facultad de Ciencias, UNAM.

**65 estados**, con narración completa y notas generadas desde `narracion.json`.
El orden se verifica contra `slides/ClasificadoresLineales.json`. Las notas del
HTML contienen este mismo texto y las indicaciones del presentador.

**Duración propuesta: 36–42 minutos.** El texto hablado contiene {words} palabras.
A 115–130 palabras por minuto, su lectura ocupa {words / 130:.0f}–{words / 115:.0f} minutos;
se añaden demostraciones, énfasis y pausas breves. Ajustar después del ensayo.
Las animaciones pueden reproducirse mientras se habla: no sumar ambas duraciones
mecánicamente. Los tiempos de MP4 siguientes son referencias del archivo sin voz,
no marcas definitivas del futuro video narrado.

## Preparar la grabación

- Abrir el HTML, usar `S` para notas y grabar únicamente la presentación.
- Avanzar al estado indicado, dejar que termine la construcción y explicar
  su fotograma estable. Durante los giros, acompañar el objeto que cambia.
- Las invitaciones a pausar dejan 2–3 segundos; no insertar minutos de silencio.
- Si se usa el MP4 en un editor, prolongar sus fotogramas estables hasta cubrir
  la narración. El archivo sólo contiene animaciones, sin pista de voz ni música.
- Mantener libres los gráficos y la columna derecha al colocar cámara o subtítulos.

## Capítulos sugeridos

| Estados | Bloque | Tiempo editorial orientativo |
|---|---|---|
| 01–13 | Geometría y perceptrón | 0–7 min |
| 14–23 | Mínimos cuadrados y contraejemplo | 7–13 min |
| 24–28 | Fisher | 13–16 min |
| 29–41 | Regresión logística y optimización | 16–25 min |
| 42–47 | Más variables y softmax | 25–29 min |
| 48–59 | Características y kernels | 29–37 min |
| 60–65 | Comparación, ejercicio y cierre | 37–42 min |

Estos tiempos son una pauta de ensayo, no capítulos de YouTube listos para publicar.

## Narración por estado

'''
start = 0.0
for i,slide in enumerate(manifest['slides'],1):
    key=f'{i:02}'
    e=entries[key]
    assert slide['notes'].startswith(f'E{key} · {e["title"]}'), (i, slide['notes'][:70])
    assert e['narration'] in slide['notes'] and e['presenter'] in slide['notes']
    file=Path(slide['file'])
    if not file.is_absolute(): file=ROOT/file
    end=start+duration(file)
    text += (f'### E{key} · {e["title"]}\n\n'
             f'**MP4 sin voz:** {timestamp(start)}–{timestamp(end)}. '
             f'**Pausa del HTML:** {i}.\n\n'
             f'{e["narration"]}\n\n'
             f'**Presentador:** {e["presenter"]}\n\n')
    start=end
text += '''## Supuestos y fuentes para preparar respuestas

La [revisión editorial](revision-youtube.md) recoge los supuestos de rango,
separación, Fisher/LDA, regularización y kernels, con referencias a los capítulos
del libro y fuentes técnicas. `validar_calculos.py` contrasta los ejemplos con
SVD y un optimizador independiente de Newton. No se deduce capacidad de
generalización de los aciertos de entrenamiento mostrados.
'''
(ROOT/'guion-youtube.md').write_text(text)
print(f'Guion sincronizado: 65 estados, {words} palabras; segmentos {start:.2f} s.')
