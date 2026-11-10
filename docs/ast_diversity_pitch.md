# AST diversity — exposición breve

## Español

Bien. Por otro lado, analizamos la diversidad del código.

La idea es simple: queremos saber si la gramática genera código diferente o si siempre genera lo mismo. Para eso comparamos los árboles sintácticos, los AST, de los operadores, generación por generación.

Medimos dos cosas.

Arriba: la distancia media entre los árboles. Cerca de 1 significa código muy diferente; cerca de 0, código muy parecido.

Abajo: la proporción de árboles únicos. Cerca de 1 significa que casi todos son distintos; cerca de 0, que hay muchas repeticiones.

Medimos esto para todos los individuos y también para la élite, los mejores de cada generación.

En ambos casos, los valores están cerca de 1: la población es diversa y la élite también. Eso indica que la gramática no colapsa a un solo código.

Es importante porque la diversidad sostiene la exploración. Actualmente buscamos mejores formas de medir diversidad en código; esta es la que usamos.

## English

Good. On the other hand, we analyzed code diversity.

The idea is simple: we want to know if the grammar generates different code or always the same code. So we compare the syntax trees, the ASTs, of the operators, generation by generation.

We measure two things.

Top: the average distance between trees. Close to 1 means very different code; close to 0 means very similar code.

Bottom: the proportion of unique trees. Close to 1 means almost all are different; close to 0 means many repeats.

We measure this for all individuals and also for the elite, the best individuals in each generation.

In both cases, the values are close to 1: the population is diverse, and the elite is diverse too. This shows that the grammar does not collapse to one code.

It matters because diversity supports exploration. We are currently looking for better ways to measure diversity in code; this is the one we used.
