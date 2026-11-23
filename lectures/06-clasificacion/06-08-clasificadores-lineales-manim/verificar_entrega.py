"""Verifica la entrega Full HD, sus notas y los bytes de sus videos incrustados."""
from pathlib import Path
from html import unescape
import base64
import hashlib
from fractions import Fraction
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
HTML=REPO/'docs/lectures/_output/06-clasificacion/06-08-clasificadores-lineales-manim/index.html'

def probe(file):
    return json.loads(subprocess.check_output(['ffprobe','-v','error',
        '-show_streams','-show_format','-of','json',str(file)],text=True))

m=json.loads((ROOT/'slides/ClasificadoresLineales.json').read_text())
entries=json.loads((ROOT/'narracion.json').read_text())
assert m['resolution']==[1920,1080]
assert len(m['slides'])==len(entries)==65
raw=HTML.read_text()
embedded=re.findall(r'data-background-video="data:video/mp4;base64,([A-Za-z0-9+/=]+)"',raw)
assert len(embedded)==65, f'Videos incrustados: {len(embedded)}'
assert not re.search(r'(?:src|href)\s*=\s*["\'](?:https?:)?//',raw), 'Recurso HTML remoto'
assert not re.search(r'url\(\s*["\']?(?:https?:)?//',raw), 'Recurso CSS remoto'
assert 'cdn.jsdelivr.net' not in raw and 'cdnjs.cloudflare.com' not in raw
notes_html=unescape(raw)
seconds=0.0
for i,(slide,video) in enumerate(zip(m['slides'],embedded),1):
    entry=entries[f'{i:02}']
    assert slide['notes'].startswith(f'E{i:02} · {entry["title"]}')
    assert entry['narration'] in slide['notes']
    assert entry['narration'] in notes_html
    file=Path(slide['file'])
    if not file.is_absolute(): file=ROOT/file
    assert hashlib.sha256(base64.b64decode(video)).digest()==hashlib.sha256(file.read_bytes()).digest()
    info=probe(file)
    streams=info['streams']
    assert len(streams)==1
    stream=streams[0]
    assert (stream['codec_name'],stream['width'],stream['height'],stream['r_frame_rate'])==('h264',1920,1080,'30/1')
    seconds+=float(info['format']['duration'])
mp4=ROOT/'media/videos/presentacion/1080p30/ClasificadoresLineales.mp4'
info=probe(mp4)
assert len(info['streams'])==1, 'El MP4 debe ser sólo animaciones, sin audio'
s=info['streams'][0]
assert (s['width'],s['height'],s['r_frame_rate'])==(1920,1080,'30/1')
assert abs(float(Fraction(s['avg_frame_rate']))-30)<0.01
assert abs(float(info['format']['duration'])-seconds)<0.2
report={'fecha':'2026-11-20','estados':65,'notas':65,'videos_incrustados':65,
        'resolucion':[1920,1080],'fps_nominal':30,'fps_promedio_mp4':float(Fraction(s['avg_frame_rate'])),'codec':'h264','audio':False,
        'duracion_mp4_segundos':float(info['format']['duration']),
        'html_bytes':HTML.stat().st_size,'html_sha256':hashlib.sha256(HTML.read_bytes()).hexdigest(),
        'mp4_sha256':hashlib.sha256(mp4.read_bytes()).hexdigest(),
        'recursos_remotos_html_css':False,'videos_identicos_al_render':True}
(ROOT/'verificacion.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
