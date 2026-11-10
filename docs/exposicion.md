# Exposición — guion breve

## 1. Diversidad AST

### Español

Primero, ¿cómo medimos la diversidad? Paso uno: cada individuo es un operador, y lo representamos como un árbol sintáctico, un AST.

Paso dos: para cada árbol, extraemos todos sus subárboles y los guardamos como huellas.

Paso tres: para dos árboles A y B, calculamos la distancia de Jaccard entre sus subárboles: uno menos la intersección dividida por la unión. Si son iguales, la distancia es cero; si no comparten nada, es uno.

Paso cuatro: para cada generación, calculamos la media de las distancias de todos los pares. Esa es la gráfica de arriba.

Paso cinco: también calculamos la proporción de árboles únicos: árboles distintos dividido por individuos. Esa es la gráfica de abajo.

Hacemos esto para todos los individuos y también para la élite, los mejores de la generación.

El objetivo es ver si la gramática produce el mismo código o código diferente.

### English

First, how do we measure diversity? Step one: each individual is an operator, and we represent it as a syntax tree, an AST.

Step two: for each tree, we extract all its subtrees and store them as fingerprints.

Step three: for two trees A and B, we compute the Jaccard distance between their subtrees: one minus the intersection divided by the union. If they are equal, the distance is zero; if they share nothing, it is one.

Step four: for each generation, we compute the average distance over all pairs. That is the top plot.

Step five: we also compute the proportion of unique trees: distinct trees divided by individuals. That is the bottom plot.

We do this for all individuals and also for the elite, the best individuals in the generation.

The goal is to see if the grammar produces the same code or different code.

## 2. Gráficas de test

### Español

Estas gráficas son de test. Cada punto es un individuo de esa generación, evaluado en cada una de las 20 instancias de test. La línea negra es la media de cada indicador.

No vemos un comportamiento suave: es bastante caótico. Hay mucha variación entre individuos y entre generaciones.

A pesar de eso, estamos generando operadores de variación competitivos frente a CX y PMX, sobre todo para MOEAD.

### English

These are the test plots. Each point is one individual from that generation, evaluated on each of the 20 test instances. The black line is the mean of each indicator.

We do not see smooth behavior: it is quite chaotic. There is a lot of variation between individuals and between generations.

Even so, we are generating variation operators that are competitive against CX and PMX, especially for MOEAD.

## 3. Qué hago

### Español

Mi trabajo es añadir un modelo de lenguaje grande a la búsqueda de G-MOHH.

El modelo propone operadores de cruce. Puede refinar operadores que ya existen o crear operadores nuevos.

Después, evaluamos esos operadores con el mismo protocolo, nos quedamos con los mejores y repetimos.

La idea es descubrir nuevos operadores de variación y compararlos de forma justa con los clásicos.

### English

My work is to add a Large Language Model to the G-MOHH search.

The model proposes crossover operators. It can refine existing operators or create new ones.

Then we evaluate those operators with the same protocol, keep the best ones, and repeat.

The idea is to discover new variation operators and compare them fairly with the classical ones.
