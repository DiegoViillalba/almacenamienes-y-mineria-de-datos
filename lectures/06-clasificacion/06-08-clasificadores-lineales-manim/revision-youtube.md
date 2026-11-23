# Revisión y storyboard para YouTube

Edición española, 20 de noviembre de 2026. Clase específica 06-08; la introducción
06-01 se consulta como referencia visual y no se modifica.

## Secuencia que se implementa

1. Pregunta concreta: si varias rectas aciertan, ¿cuál elegimos? Geometría antes de notación.
2. Separar, girar w y deslizar b; distancia con signo; perceptrón con tres correcciones.
3. Los seis estudiantes: residuos → ajuste → cálculo manual → ecuaciones normales.
4. Contraejemplo: observaciones correctas y lejanas cambian MCO.
5. Fisher: proyección → giro → criterio → dirección → supuesto del umbral.
6. Logística: puntaje → sigmoide → pérdida individual → entrenamiento → optimización.
7. Existencia del óptimo y regularización; volver al contraejemplo sin prometer robustez general.
8. Niveles paralelos → proyección sobre w → tres variables → softmax y empates activos.
9. Dos cortes en una variable → elevar → volver; anillo; kernel numérico; RBF y su objetivo.
10. Comparación por criterios; ejercicio con tres revelaciones; cierre con transferencia.

Cada derivación se reparte en pausas con una sola operación nueva. Los puntos y
sus formas se conservan en las transformaciones. Las notas y el guion distinguen
predicción, animación y explicación del estado estable.

## Correcciones conceptuales

- Hiperplano con w distinto de cero; d+1 coeficientes con redundancia de escala.
- Infinitos separadores para esta muestra finita con margen estricto; no afirmar que casi todos los ángulos separan.
- MCO: inversa sólo con rango completo; en práctica resolver con QR/SVD.
- Fisher: S_W definida positiva; S_B = Δm Δmᵀ. La dirección no fija por sí sola el umbral.
  Punto medio con probabilidades previas y costos iguales; interpretación LDA con gaussianas y covarianza común.
- Logística: convexidad no garantiza unicidad ni existencia; rango completo y ausencia de separación completa/cuasi completa.
  Newton puede requerir amortiguación y gradiente una tasa adecuada; los conteos son de este ejemplo.
- Log-loss promedio con penalización λ||w||²/2 y λ=0.01 en el ejemplo separable de puntos lejanos.
  Menor influencia de puntos correctos muy confiados no implica resistencia general a atípicos.
- Softmax: los empates entre clases sólo forman frontera visible si superan a las demás; son porciones de hiperplanos.
- Kernel ridge: minimizar ||y−Kα||² + λ αᵀKα, sin intercepto en este ejemplo.
  Lineal en φ no significa frontera lineal en x ni que cualquier algoritmo acepte sustituir productos punto.
- Las cifras son de entrenamiento e ilustran criterios; no establecen cuál método generaliza mejor.

## Fuentes de contraste

Fuentes del libro: capítulos de clasificadores lineales, logística y SVM.
Además: [modelos lineales](https://scikit-learn.org/stable/modules/linear_model.html),
[LDA](https://scikit-learn.org/stable/modules/lda_qda.html) y
[kernel ridge](https://scikit-learn.org/stable/modules/kernel_ridge.html).
Estas fuentes apoyan la distinción entre objetivo, supuestos y forma de la frontera.

## Registro de validación

Entrega completada el 20 de noviembre de 2026.

- Tres iteraciones de prueba a 854 × 480 y 15 fps, seguidas de render completo
  a 1920 × 1080 y 30 fps. El pase editorial final reutilizó caché de Manim;
  no se reutilizó freeze del libro.
- 65 estados y 65 notas; guion de 4.220 palabras sincronizado con IDs E01–E65
  y los tiempos de los segmentos finales. Duración narrada propuesta: 36–42 min.
- MP4 H.264 de 319.18 s, sin pista de audio. FPS nominal 30; el promedio que
  reporta ffprobe es 30.002 por las marcas temporales de concatenación de Manim.
- HTML único de 29.781.698 bytes (unos 28.4 MiB), con los 65 segmentos idénticos
  al render, notas completas y sin referencias remotas en recursos HTML/CSS.
  `verificacion.json` conserva metadatos y SHA-256 de ambas entregas.
- Inspección de los 65 fotogramas finales y 63 muestras de 21 transiciones:
  giro y desplazamiento de fronteras, tres correcciones del perceptrón, MCO,
  llegada de lejanos, Fisher, sigmoide, entrenamiento, optimizadores, proyección,
  cubo, elevación, anillo y campanas RBF. Se repitió el pase de los estados
  modificados al cerrar la edición.
- Se corrigieron recortes de texto, rótulos superpuestos, una pausa redundante,
  curvas de nivel deformadas por recorte artificial, polígonos que se deformaban
  al interpolar vértices y un aumento transitorio de SSE. Ahora las fronteras y
  regiones se reconstruyen desde los coeficientes en cada fotograma; MCO se mueve
  directamente hacia el mínimo. Los marcadores permanecen sobre las regiones.
- Verificación independiente: MCO/SVD, logística/BFGS, actualizaciones del
  perceptrón, criterio de Fisher, fronteras activas de softmax, productos y
  sistema kernel ridge, y ejercicio. Compilación Python y `git diff --check`.
- En navegador local se comprobó avance, retroceso, vista general, reproducción
  y presencia de 65 notas en el DOM, sin errores de consola. El navegador
  integrado no mostró una ventana adicional con S: se verificó el texto
  incrustado; la ventana del presentador queda sin confirmar en ese navegador.
  El guion separado permite narrar sin depender de esa ventana.
- Galería actualizada y renderizada con `--to html --no-execute`. No se ejecutaron
  capítulos, no se actualizó freeze ni se regeneró el PDF del libro. No hubo push
  ni publicación. Se comprobaron por SHA-256 los cambios preexistentes ajenos;
  siguen intactos. La edición previa de la tarjeta 06-01 en la galería se conserva.

Evidencia local reproducible (ignorada por Git): `media/qa/fullhd/` contiene
fotogramas y hojas de contacto; `media/qa/transitions/` conserva muestras
intermedias y el pase de correcciones. El HTML y el MP4 contienen animaciones
sin voz: no constituyen un video narrado terminado.
