# P* con N = poblacion de MOEAD

Generado con `generate_reference_sets_pop.py` (runs=21, iterations=10000, select=spread, MO_POPULATION_SIZE=105).

El tamano pedido es el de la poblacion real de MOEAD para el numero de objetivos de la instancia (Das-Dennis): 105 para m=2 y m=3, 120 para m=4 y 126 para m=5.

| split | instancia | m | N pedido | N escrito | ground | truncado | run |
|---|---|---:|---:|---:|---:|:---:|---:|
| test | KC100-5fl-1un | 5 | 126 | 44 | 44 | si | 1 |
| test | KC11-2fl-1rl | 2 | 105 | 38 | 38 | si | 1 |
| test | KC11-2fl-1un | 2 | 105 | 24 | 24 | si | 1 |
| test | KC12-5fl-1rl | 5 | 126 | 126 | 6212 | no | 12 |
| test | KC18-4fl-1un | 4 | 120 | 15 | 15 | si | 1 |
| test | KC21-5fl-1un | 5 | 126 | 126 | 15941 | no | 2 |
| test | KC31-3fl-1un | 3 | 105 | 105 | 2092 | no | 17 |
| test | KC33-4fl-1un | 4 | 120 | 120 | 3474 | no | 21 |
| test | KC38-4fl-1rl | 4 | 120 | 120 | 1020 | no | 8 |
| test | KC39-4fl-1un | 4 | 120 | 120 | 3345 | no | 5 |
| test | KC44-5fl-1un | 5 | 126 | 126 | 14794 | no | 10 |
| test | KC45-2fl-1un | 2 | 105 | 25 | 25 | si | 1 |
| test | KC46-4fl-1rl | 4 | 120 | 120 | 1022 | no | 16 |
| test | KC51-2fl-1un | 2 | 105 | 104 | 104 | si | 1 |
| test | KC59-2fl-1rl | 2 | 105 | 105 | 155 | no | 2 |
| test | KC63-3fl-1un | 3 | 105 | 86 | 86 | si | 1 |
| test | KC64-2fl-1rl | 2 | 105 | 105 | 178 | no | 4 |
| test | KC76-5fl-1rl | 5 | 126 | 126 | 1648 | no | 17 |
| test | KC77-2fl-1un | 2 | 105 | 12 | 12 | si | 1 |
| test | KC82-3fl-1rl | 3 | 105 | 105 | 438 | no | 9 |
| train | KC100-3fl-1rl | 3 | 105 | 105 | 236 | no | 14 |
| train | KC13-3fl-1un | 3 | 105 | 24 | 24 | si | 1 |
| train | KC14-3fl-1rl | 3 | 105 | 105 | 1847 | no | 5 |
| train | KC15-4fl-1un | 4 | 120 | 120 | 5905 | no | 20 |
| train | KC16-4fl-1rl | 4 | 120 | 120 | 7524 | no | 16 |
| train | KC16-4fl-1un | 4 | 120 | 8 | 8 | si | 1 |
| train | KC17-3fl-1rl | 3 | 105 | 105 | 1389 | no | 20 |
| train | KC20-4fl-1rl | 4 | 120 | 120 | 4307 | no | 5 |
| train | KC20-4fl-1un | 4 | 120 | 120 | 5039 | no | 6 |
| train | KC22-3fl-1rl | 3 | 105 | 105 | 848 | no | 9 |
| train | KC23-2fl-1rl | 2 | 105 | 105 | 106 | no | 2 |
| train | KC23-3fl-1un | 3 | 105 | 105 | 760 | no | 6 |
| train | KC24-5fl-1rl | 5 | 126 | 126 | 6752 | no | 7 |
| train | KC26-2fl-1un | 2 | 105 | 105 | 165 | no | 6 |
| train | KC27-4fl-1un | 4 | 120 | 120 | 4232 | no | 4 |
| train | KC27-5fl-1rl | 5 | 126 | 126 | 3116 | no | 14 |
| train | KC28-3fl-1un | 3 | 105 | 105 | 271 | no | 1 |
| train | KC29-5fl-1rl | 5 | 126 | 126 | 1136 | no | 6 |
| train | KC31-4fl-1rl | 4 | 120 | 120 | 925 | no | 18 |
| train | KC32-4fl-1rl | 4 | 120 | 120 | 2018 | no | 9 |
| train | KC33-3fl-1un | 3 | 105 | 7 | 7 | si | 1 |
| train | KC35-3fl-1rl | 3 | 105 | 105 | 464 | no | 19 |
| train | KC35-5fl-1rl | 5 | 126 | 126 | 2338 | no | 21 |
| train | KC36-3fl-1un | 3 | 105 | 105 | 783 | no | 4 |
| train | KC38-5fl-1un | 5 | 126 | 126 | 8932 | no | 9 |
| train | KC40-5fl-1rl | 5 | 126 | 126 | 978 | no | 21 |
| train | KC41-4fl-1rl | 4 | 120 | 120 | 1324 | no | 6 |
| train | KC42-3fl-1un | 3 | 105 | 67 | 67 | si | 1 |
| train | KC44-3fl-1rl | 3 | 105 | 105 | 444 | no | 2 |
| train | KC47-4fl-1rl | 4 | 120 | 120 | 290 | no | 4 |
| train | KC48-2fl-1un | 2 | 105 | 105 | 251 | no | 14 |
| train | KC48-4fl-1un | 4 | 120 | 120 | 3645 | no | 6 |
| train | KC50-2fl-1rl | 2 | 105 | 105 | 151 | no | 21 |
| train | KC50-3fl-1rl | 3 | 105 | 105 | 391 | no | 10 |
| train | KC51-4fl-1rl | 4 | 120 | 120 | 543 | no | 15 |
| train | KC52-5fl-1un | 5 | 126 | 126 | 1521 | no | 7 |
| train | KC55-3fl-1un | 3 | 105 | 71 | 71 | si | 1 |
| train | KC55-5fl-1rl | 5 | 126 | 126 | 835 | no | 20 |
| train | KC56-2fl-1un | 2 | 105 | 18 | 18 | si | 1 |
| train | KC56-3fl-1rl | 3 | 105 | 105 | 375 | no | 13 |
| train | KC58-3fl-1rl | 3 | 105 | 105 | 489 | no | 18 |
| train | KC59-4fl-1un | 4 | 120 | 120 | 557 | no | 19 |
| train | KC60-4fl-1un | 4 | 120 | 120 | 1030 | no | 19 |
| train | KC61-3fl-1un | 3 | 105 | 34 | 34 | si | 1 |
| train | KC61-4fl-1rl | 4 | 120 | 120 | 448 | no | 21 |
| train | KC64-2fl-1rl | 2 | 105 | 105 | 127 | no | 3 |
| train | KC64-2fl-1un | 2 | 105 | 105 | 140 | no | 5 |
| train | KC67-4fl-1rl | 4 | 120 | 120 | 678 | no | 4 |
| train | KC68-4fl-1un | 4 | 120 | 120 | 4903 | no | 6 |
| train | KC68-4fl-2un | 4 | 120 | 41 | 41 | si | 1 |
| train | KC69-3fl-1rl | 3 | 105 | 105 | 347 | no | 19 |
| train | KC71-2fl-1rl | 2 | 105 | 105 | 199 | no | 7 |
| train | KC71-3fl-1un | 3 | 105 | 71 | 71 | si | 1 |
| train | KC71-4fl-1rl | 4 | 120 | 120 | 840 | no | 10 |
| train | KC72-5fl-1un | 5 | 126 | 126 | 8614 | no | 3 |
| train | KC74-3fl-1rl | 3 | 105 | 105 | 349 | no | 10 |
| train | KC74-4fl-1un | 4 | 120 | 120 | 380 | no | 16 |
| train | KC76-3fl-1un | 3 | 105 | 105 | 448 | no | 12 |
| train | KC77-3fl-1rl | 3 | 105 | 105 | 466 | no | 7 |
| train | KC79-2fl-1rl | 2 | 105 | 105 | 144 | no | 16 |
| train | KC79-3fl-1un | 3 | 105 | 41 | 41 | si | 1 |
| train | KC80-5fl-1un | 5 | 126 | 126 | 6989 | no | 12 |
| train | KC83-5fl-1un | 5 | 126 | 126 | 623 | no | 19 |
| train | KC84-3fl-1rl | 3 | 105 | 105 | 260 | no | 9 |
| train | KC84-3fl-1un | 3 | 105 | 23 | 23 | si | 1 |
| train | KC84-4fl-1rl | 4 | 120 | 120 | 1907 | no | 19 |
| train | KC87-3fl-1un | 3 | 105 | 105 | 407 | no | 10 |
| train | KC87-4fl-1rl | 4 | 120 | 120 | 928 | no | 19 |
| train | KC88-4fl-1un | 4 | 120 | 120 | 3681 | no | 19 |
| train | KC88-5fl-1rl | 5 | 126 | 126 | 1874 | no | 5 |
| train | KC90-4fl-1rl | 4 | 120 | 120 | 1087 | no | 13 |
| train | KC90-4fl-1un | 4 | 120 | 30 | 30 | si | 1 |
| train | KC92-3fl-1rl | 3 | 105 | 105 | 266 | no | 13 |
| train | KC93-4fl-1rl | 4 | 120 | 120 | 657 | no | 13 |
| train | KC93-4fl-1un | 4 | 120 | 120 | 386 | no | 18 |
| train | KC95-3fl-1un | 3 | 105 | 48 | 48 | si | 1 |
| train | KC95-4fl-1un | 4 | 120 | 120 | 3667 | no | 8 |
| train | KC96-3fl-1rl | 3 | 105 | 105 | 182 | no | 21 |
| train | KC97-3fl-1rl | 3 | 105 | 105 | 161 | no | 10 |
| train | KC97-3fl-1un | 3 | 105 | 30 | 30 | si | 1 |

Instancias truncadas (|ground| < N): 22
- test/KC100-5fl-1un: 44 < 126
- test/KC11-2fl-1rl: 38 < 105
- test/KC11-2fl-1un: 24 < 105
- test/KC18-4fl-1un: 15 < 120
- test/KC45-2fl-1un: 25 < 105
- test/KC51-2fl-1un: 104 < 105
- test/KC63-3fl-1un: 86 < 105
- test/KC77-2fl-1un: 12 < 105
- train/KC13-3fl-1un: 24 < 105
- train/KC16-4fl-1un: 8 < 120
- train/KC33-3fl-1un: 7 < 105
- train/KC42-3fl-1un: 67 < 105
- train/KC55-3fl-1un: 71 < 105
- train/KC56-2fl-1un: 18 < 105
- train/KC61-3fl-1un: 34 < 105
- train/KC68-4fl-2un: 41 < 120
- train/KC71-3fl-1un: 71 < 105
- train/KC79-3fl-1un: 41 < 105
- train/KC84-3fl-1un: 23 < 105
- train/KC90-4fl-1un: 30 < 120
- train/KC95-3fl-1un: 48 < 105
- train/KC97-3fl-1un: 30 < 105

## Validacion

Comprobaciones: cabecera `# N m` coherente, todo punto de P* presente en el ground set, sin puntos dominados dentro de P* y cardinalidad igual a N (o al ground set si este es menor).

- train/KC79-3fl-1un: el ground set ya contiene puntos dominados y P* los hereda (preexistente)
