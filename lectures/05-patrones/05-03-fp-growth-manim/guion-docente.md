# Guion docente · FP-Growth: comprimir, condicionar y crecer

**Duración:** 60 minutos. **Fuentes conceptuales:**
`unidades/reglas_asociacion/5-3-fp_growth.qmd` (ejemplo base, definiciones,
laboratorio y comparación con Apriori) y `5-1-apriori.qmd` (las 50 canastas).
Clase previa: `05-02-reglas-asociacion-manim` (soporte, confianza, lift).

La presentación (`index.html`) tiene una nota en cada estado: la tecla `S` abre
la vista del presentador. Este guion organiza el tiempo, los cálculos de
pizarra, las respuestas esperadas y los errores que conviene anticipar.
Todos los números en pantalla se calculan en `sections/data.py`
(`python -m sections.data` imprime el resumen usado aquí y comprueba que
FP-Growth coincide con la fuerza bruta en los tres conjuntos de datos).

## Idea que recorre la clase

> Apriori gasta su tiempo en fabricar candidatos y releer la base por nivel.
> FP-Growth lee la base dos veces: una para contar y ordenar, otra para
> escribir los tickets como palabras en un árbol de prefijos. Después parte el
> problema por el último ítem de cada patrón y resuelve cada parte dentro de una
> base condicional: los caminos prefijo, pesados con el contador del ítem.

La clase avanza de lo cotidiano a lo formal: el costo de Apriori que ya
conocen → la analogía de las palabras con prefijo común → el FP-tree construido
a mano → el minado guiado (Cerveza) y autónomo (Pañales) → pseudocódigo →
comparación, escala y Python → transferencia con datos nuevos.

## Objetivos verificables

Al salir, el estudiante podrá:

1. Explicar con números por qué Apriori se vuelve caro: candidatos por nivel y
   una lectura completa de la base por nivel.
2. Construir un FP-tree a mano: conteos, poda, orden global con desempate,
   inserción y tabla de encabezados.
3. Obtener la base condicional de un ítem con los pesos correctos, podarla,
   construir el árbol condicional y enumerar los patrones.
4. Decidir cuándo conviene FP-Growth frente a Apriori y ejecutarlo con
   `mlxtend`, sabiendo que las reglas se generan después.

**Material:** proyector, pizarra, hojas para dibujar árboles (el «tu turno» y
el reto se hacen en papel).

## Secuencia minuto a minuto

| Min. | Acto / estados | Acción docente | Evidencia de aprendizaje |
|---|---|---|---|
| 0–3 | Portada y objetivos | Preguntar qué pasaría con Apriori en una tienda de 10 000 productos. | Intuiciones sobre el costo (anotarlas). |
| 3–10 | **Intuición** | Apriori nivel por nivel con contador de lecturas; la cuenta de pares; las palabras sol, sola, solar, sopa en un árbol; «un ticket no tiene orden». | Dicen que el costo está en los candidatos y las relecturas; votan qué orden comprime más. |
| 10–24 | **1 · Comprimir** | Conteos y umbral; orden global; reordenar tickets; insertar T1…T5 con predicción; leer el árbol; tabla de encabezados; orden inverso (10 vs 6 nodos); definición. | Predicen nodos nuevos por ticket; explican por qué T3 abre rama; suman 3 + 1 para Pañales. |
| 24–34 | **2 · Minar (guiado)** | Partición 8 + 4 + 2 + 1; enlaces de Cerveza; predicción del peso; base condicional; poda de Pan; árbol condicional y 4 patrones. | Votan el peso (respuesta: 2); explican por qué Pan sale. |
| 34–38 | **Tu turno** | Pañales en parejas (3 min) y solución. | Base [Leche, Pan] × 3, [Leche] × 1; cuatro patrones. |
| 38–42 | **2 · Minar (cierre)** | Pan y Leche; balance de 11 de 15; recursión con mínimo 2; pseudocódigo. | Explican que los 4 infrecuentes nunca fueron candidatos. |
| 42–50 | **3 · Comparar** | Apriori vs FP-Growth en el ejemplo; curva de las 50 canastas y datos densos; tabla de cuándo conviene; código `mlxtend`. | Relacionan compresión con similitud entre tickets; convierten 3/5 a soporte relativo. |
| 50–57 | **Reto en parejas** | Seis tickets de cafetería: orden, árbol, minar azúcar y chocolate. | Resultados del reto (abajo). |
| 57–60 | Cierre y ticket | Cuatro verbos: contar, comprimir, condicionar, crecer. Ticket individual. | Respuesta escrita del ticket. |

### Si el tiempo apremia

- En la intuición, saltar la tabla de pares (min 5–6) y decir sólo la cifra de
  10 000 productos (−1 min).
- En «Comprimir», insertar T4 y T5 sin pausa de predicción (−1.5 min) y omitir
  la diapositiva de formalización, que repite los cinco pasos (−1 min).
- En «Comparar», pasar la tabla de cuándo conviene en 30 segundos (−1.5 min).
- No recortar la predicción del peso de Leche (min 28), el «tu turno» ni el
  reto: son la evidencia principal de comprensión.

## Hilo conductor y formalización

### Intuición (min 3–10)

**Apriori sobre los cinco tickets del libro** (soporte mínimo 3):

| Nivel | Candidatos contados | Podados sin contar | Frecuentes |
|---|---|---|---|
| 1 | 4 | 0 | 4 |
| 2 | 6 | 0 | 5 ({Pan, Cerveza} = 2 cae) |
| 3 | 2 | 2 ({Leche, Pan, Cerveza}, {Pan, Pañales, Cerveza}) | 2 |
| 4 | 0 | 1 ({Leche, Pan, Pañales, Cerveza}) | 0 |

Total: **3 lecturas** de la base y **12 candidatos** contados. Pregunta de
control: con un patrón frecuente de 10 ítems, ¿cuántas lecturas? Al menos 10,
una por nivel.

**La cuenta que no escala.** Pares candidatos = m(m − 1)/2: 4 → 6,
100 → 4 950, 1 000 → 499 500, 10 000 → 49 995 000. Un patrón frecuente de 20
ítems implica 2²⁰ − 1 = 1 048 575 subconjuntos frecuentes (antimonotonía), y
Apriori fabrica cada uno como candidato.

**Palabras con prefijo común.** sol, sola, solar, sopa: 16 letras escritas,
7 nodos en el árbol (s:4 → o:4 → l:3 → a:2 → r:1, y o → p:1 → a:1). Cada
número dice cuántas palabras pasan por el nodo; ninguna palabra se pierde.

**El obstáculo.** Un ticket es un conjunto: {Pan, Leche} = {Leche, Pan}. Hay
que *elegir* un orden para escribirlo como palabra. Se vota aquí y se
resuelve en el minuto 22.

### 1 · Comprimir (min 10–24)

Tickets (min_sup = 3): T1 = {Pan, Leche}, T2 = {Pan, Leche, Pañales, Cerveza},
T3 = {Leche, Pañales, Cerveza}, T4 = {Pan, Leche, Pañales, Cerveza},
T5 = {Pan, Leche, Pañales}.

1. **Lectura 1 · contar:** Leche 5, Pan 4, Pañales 4, Cerveza 3. Nadie cae.
2. **Orden global:** Leche ≻ Pan ≻ Pañales ≻ Cerveza (empate Pan = Pañales,
   desempate alfabético; lo importante es que la regla sea fija).
3. **Lectura 2 · insertar** (predicción antes de cada uno):

| Ticket ordenado | Comparte | Crea | Comentario |
|---|---|---|---|
| [Leche, Pan] | 0 | 2 | primer camino |
| [Leche, Pan, Pañales, Cerveza] | 2 | 2 | extiende la rama |
| [Leche, Pañales, Cerveza] | 1 | 2 | **única** rama nueva bajo Leche |
| [Leche, Pan, Pañales, Cerveza] | 4 | 0 | repetido: cero nodos |
| [Leche, Pan, Pañales] | 3 | 0 | sólo contadores |

Árbol final: Leche:5 → Pan:4 → Pañales:3 → Cerveza:2, y Leche → Pañales:1 →
Cerveza:1. **16 apariciones → 6 nodos** (38 %).

- **Lectura de un contador:** Leche → Pan: 4 = cuatro tickets *empiezan* con
  [Leche, Pan]. Al bajar, los contadores no crecen.
- **Soporte de un ítem = suma de sus nodos:** Pañales 3 + 1 = 4. Por eso la
  tabla de encabezados enlaza los nodos de cada ítem.
- **Orden inverso** (Cerveza ≻ Pan ≻ Pañales ≻ Leche): 10 nodos contra 6. Los
  ítems frecuentes arriba son prefijo de más tickets.
- **Formalización:** count(v) = número de tickets ordenados cuyo prefijo es
  el camino raíz → v. Peor caso: sin prefijos compartidos, un nodo por
  aparición más los punteros; el árbol puede ocupar más memoria que la lista.

### 2 · Minar (min 24–42)

**Partición.** Con el orden global, cada itemset tiene un último ítem: 8
terminan en Cerveza, 4 en Pañales (sin Cerveza), 2 en Pan, 1 en Leche. Los
subproblemas son disjuntos y cubren todo. Se empieza por el menos frecuente
porque sus caminos prefijo contienen a todos sus posibles acompañantes; al
pasar a Pañales, los nodos Cerveza que cuelgan debajo ya no importan.

**Cerveza (guiado).**

| Paso | Resultado |
|---|---|
| Enlaces | dos nodos, 2 + 1 = 3 ⇒ {Cerveza}: 3 |
| Base condicional | [Leche, Pan, Pañales] × 2, [Leche, Pañales] × 1 |
| Conteos condicionales | Leche 3, Pan 2 ✗, Pañales 3 |
| Árbol condicional | Leche:3 → Pañales:3 (un solo camino) |
| Patrones | {Cerveza}, {Leche, Cerveza}, {Pañales, Cerveza}, {Leche, Pañales, Cerveza}: todos 3 |

La predicción clave: **Leche entra con peso 2, no 5.** El peso de cada camino
prefijo es el contador del nodo Cerveza al final del camino. Conecte con la
clase pasada: es condicionar, el universo se encoge a los tickets con Cerveza.

**Pañales (tu turno).** Nodos Pañales:3 (camino Leche → Pan) y Pañales:1
(camino Leche). Base [Leche, Pan] × 3, [Leche] × 1; conteos Leche 4, Pan 3;
árbol Leche:4 → Pan:3; patrones {Pañales}: 4, {Leche, Pañales}: 4,
{Pan, Pañales}: 3, {Leche, Pan, Pañales}: 3.

**Pan y Leche.** Pan: base [Leche] × 4 ⇒ {Pan}: 4, {Leche, Pan}: 4. Leche
cuelga de la raíz: base vacía ⇒ {Leche}: 5.

**Balance.** 11 de los 15 itemsets son frecuentes (coincide con el libro).
Los 4 restantes contienen Pan y Cerveza; desaparecieron al podar Pan en la
base de Cerveza y nunca se contaron.

**Recursión.** Con soporte mínimo 2, Pan se queda en la base de Cerveza y el
árbol condicional se ramifica: Leche:3 → Pan:2 → Pañales:2, y
Leche → Pañales:1. Se aplica el mismo procedimiento dentro de ese árbol. En la
pantalla, cada nodo del árbol de llamadas es un patrón (léase subiendo hasta
Cerveza); salen 8 patrones con Cerveza: {Cerveza}: 3, {Pañales, Cerveza}: 3,
{Pan, Pañales, Cerveza}: 2, {Leche, Pan, Pañales, Cerveza}: 2,
{Leche, Pañales, Cerveza}: 3, {Pan, Cerveza}: 2, {Leche, Pan, Cerveza}: 2 y
{Leche, Cerveza}: 3.

**Formalización.**
`B_a = {(P_j, c_j) : P_j camino prefijo de un nodo a, c_j = count(ese nodo a)}`
y el pseudocódigo de siete líneas: recorrer la tabla de abajo hacia arriba,
reportar {a} ∪ sufijo con la suma de contadores de a, construir la base, podar
y llamar de nuevo sobre el FP-tree de la base.

### 3 · Comparar (min 42–50)

| | Apriori | FP-Growth |
|---|---|---|
| Lecturas de la base (ejemplo) | 3 | 2 |
| Candidatos contados (ejemplo) | 12 (3 más podados) | 0 |
| Estructuras | listas de candidatos | árbol de 6 nodos + 3 bases condicionales |
| Resultado | 11 itemsets | los mismos 11 |

**A escala, 50 canastas del capítulo de Apriori** (mínimo 5 canastas): 146
apariciones de ítems → 59 nodos (40 %); 26 itemsets frecuentes; Apriori cuenta
98 candidatos en 3 lecturas, FP-Growth hace 2 lecturas. La curva rosa son 50
canastas **sintéticas** densas (cada una es un prefijo de una misma lista de
5 productos): 150 apariciones → 5 nodos. Mensaje: la compresión depende de
cuánto se parecen los tickets, no del algoritmo.

**Cuándo conviene:** FP-Growth en bases grandes y densas; Apriori para
explicar, para bases pequeñas o dispersas, o cuando se quiere inspeccionar
cada nivel. Límites de FP-Growth: memoria del árbol, implementación más
compleja, no es incremental, y un umbral bajo sigue produciendo muchísimos
patrones.

**Python:** `fpgrowth(df, min_support=3/5, use_colnames=True)` sobre la misma
matriz one-hot de `TransactionEncoder`; `min_support` es relativo. Las reglas
se obtienen después con `association_rules` (confianza, lift), como en la
clase anterior.

## Reto en parejas (min 50–57)

Seis tickets, soporte mínimo 3: {galletas, café, chocolate},
{azúcar, café, galletas}, {chocolate, azúcar, café}, {galletas, chocolate},
{café, azúcar, chocolate, galletas}, {té, galletas, café}.

| Paso | Resultado |
|---|---|
| Conteos | café 5, galletas 5, chocolate 4, azúcar 3, **té 1 (sale)** |
| Orden | café ≻ galletas ≻ chocolate ≻ azúcar (empate café = galletas, alfabético) |
| Árbol | café:5 → galletas:4 → chocolate:2 → azúcar:1; galletas:4 → azúcar:1; café → chocolate:1 → azúcar:1; ∅ → galletas:1 → chocolate:1. **17 apariciones → 9 nodos** |
| Azúcar | base [café, galletas] × 1, [café, chocolate] × 1, [café, galletas, chocolate] × 1; café 3, galletas 2 ✗, chocolate 2 ✗ ⇒ {azúcar}: 3, {café, azúcar}: 3 |
| Chocolate | base [café, galletas] × 2, [café] × 1, [galletas] × 1; café 3, galletas 3 ⇒ {chocolate}: 4, {café, chocolate}: 3, {galletas, chocolate}: 3 |
| Trampa | {café, galletas, chocolate} sólo en 2 tickets (T1, T5): **no** es frecuente. El árbol condicional de chocolate se ramifica (café:3 → galletas:2 y galletas:1); dentro de él, galletas tiene base [café] × 2 |

Itemsets frecuentes del reto (8): {café}: 5, {galletas}: 5, {chocolate}: 4,
{azúcar}: 3, {café, galletas}: 4, {café, chocolate}: 3,
{galletas, chocolate}: 3, {café, azúcar}: 3.

**Pista si se atoran:** «¿qué peso tiene cada camino?» (el del nodo del ítem
que se mina). **Pregunta de control:** *¿por qué café y galletas llegan a 3 en
la base de chocolate pero el trío no?* Porque cada uno llega por caminos
distintos; sólo el camino [café, galletas] × 2 los tiene juntos.

## Errores frecuentes a vigilar

- **Copiar los contadores del prefijo** (Leche × 5) en vez del contador del
  nodo minado (× 2). Es el error más común; por eso se vota antes de revelar.
- **Reusar un nodo de otra rama.** T3 no puede sumar al Pañales que cuelga de
  Pan: el camino desde la raíz es distinto.
- **Olvidar podar antes de construir** (el té del reto) o no fijar el
  desempate: el árbol cambia de forma y deja de coincidir entre equipos.
- **Incluir los nodos que cuelgan debajo** del ítem minado (Cerveza al minar
  Pañales). Esos patrones pertenecen a otra columna de la partición.
- **Concluir que un trío es frecuente porque sus pares lo son** (café,
  galletas, chocolate). La antimonotonía sólo funciona en un sentido.
- **Pensar que FP-Growth da otro resultado o reglas directamente.** Da los
  mismos itemsets que Apriori; las reglas se calculan después.

## Ticket de salida (individual)

> 1. El único nodo Huevos está en el camino Leche: 6 → Pan: 4 → Huevos: 2.
>    ¿Qué entra a la base condicional de Huevos?
> 2. ¿Por qué se insertan primero los ítems más frecuentes?

**Respuestas esperadas:** (1) [Leche, Pan] × 2: el peso es el contador del
nodo Huevos. (2) Porque los ítems frecuentes son prefijo de más tickets; al
ir arriba se comparten y el árbol tiene menos nodos (10 contra 6 en el
ejemplo).

## Notas sobre las fuentes y convenciones

- Los árboles condicionales conservan el **orden global**, como el
  laboratorio del capítulo 5.3. El código de ese capítulo reordena por
  conteo condicional; ambos criterios producen exactamente los mismos
  itemsets frecuentes, sólo cambia la forma del árbol condicional.
- El ejemplo base, los 11 itemsets y la base de Cerveza coinciden con el
  capítulo 5.3; los 6 nodos / 16 apariciones coinciden con los valores de
  referencia de su laboratorio.
- La curva de datos densos usa canastas sintéticas construidas en
  `sections/data.py` (`DENSE`); se presentan como tales en pantalla.
