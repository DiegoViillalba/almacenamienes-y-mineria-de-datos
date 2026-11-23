# Guion docente · Clasificadores lineales

La edición compilada es la de grabación. El [guion completo](guion-youtube.md)
contiene la narración de sus 65 estados, las notas del presentador y los tiempos
referidos al MP4 sin voz. `narracion.json` es la fuente compartida con las notas
del HTML; `sincronizar_guion.py` verifica su correspondencia con el render.

## Uso en clase (60–70 minutos con participación)

| Estados | Actividad | Minutos sugeridos |
|---|---|---:|
| 01–13 | Predecir una recta; leer w y b; seguir el perceptrón | 12 |
| 14–23 | Verificar sumas manuales y anticipar el efecto de lejanos | 12 |
| 24–28 | Proponer dirección y distinguirla del umbral de Fisher | 7 |
| 29–41 | Interpretar log-loss, contrastar métodos y regularizar | 16 |
| 42–47 | Derivar niveles de p y reconocer empates activos de softmax | 7 |
| 48–59 | Regresar desde φ; verificar el producto kernel | 10 |
| 60–65 | Resolver P, Q y R y recuperar el criterio de elección | 6 |

En el aula puede dejarse más tiempo en E17, E22 y E61. Para grabar, invitar a
pausar el video y continuar tras unos segundos. Conservar todos los supuestos
de E09, E20, E28, E38, E41 y E58: no son ampliaciones prescindibles.

## Respuestas del ejercicio

Con w=(2,−1), b=−3: P=(3,1) da z=2 y p≈0.88; Q=(1,2) da z=−3 y p≈0.05;
R=(2,1) da z=0 y p=0.5. El último está sobre la frontera y requiere una
convención binaria de desempate. Duplicar w y b conserva la frontera y hace más
extremas las probabilidades de P y Q; R permanece en 0.5.

## Aclaraciones que conviene anticipar

- Convexidad sola no garantiza existencia ni unicidad del óptimo logístico.
- MCO usa una inversa sólo con rango completo; resolver numéricamente con QR/SVD.
- Fisher define una dirección; el umbral medio supone previas y costos iguales.
- Softmax tiene porciones activas de hiperplanos, no un único hiperplano global.
- RBF es no lineal en los atributos originales; especificar el espacio al decir lineal.
- El ejemplo RBF es kernel ridge sin intercepto, no una SVM entrenada ni probabilidades.
- Los aciertos son de entrenamiento; no justifican una recomendación de generalización.

La [revisión editorial](revision-youtube.md) detalla los cambios y sus fuentes.
