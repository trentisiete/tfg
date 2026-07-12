
# Diapositiva 14 — Resultados sintéticos: mejora y ritmo de búsqueda

## Mensaje de la diapositiva

**GP + EI mejora el \emph{incumbent} en los cuatro benchmarks, pero no todos los problemas mejoran con la misma intensidad ni al mismo ritmo.**

---

## Título

**Resultados sintéticos: mejora y ritmo de búsqueda**

## Subtítulo

**EI mejora el mejor valor encontrado en los cuatro benchmarks; la dificultad del problema condiciona tanto la mejora final como cuándo aparece.**

---

## Visual principal

Gráfico de barras con estos datos:

| Benchmark | Mejor modelo SBO   | Mejora final | Ritmo |
| --------- | ------------------ | -----------: | ----: |
| Forrester | GP Matérn 5/2     |        0.737 | 0.530 |
| Branin    | GP Matérn 5/2 ARD |        0.898 | 0.663 |
| Hartmann6 | GP RBF             |        0.434 | 0.701 |
| Borehole  | GP Linear          |        0.720 | 0.963 |

La mejora final corresponde a la mejora relativa del mínimo encontrado al terminar el proceso de \emph{infill}; en Forrester, Branin y Hartmann6 equivale a reducción relativa del gap al óptimo, mientras que en Borehole se usa la mejora relativa del mejor valor limpio respecto al diseño inicial.

### Etiquetas sobre las barras

* **Forrester:** `0.737 · ritmo 0.53`
* **Branin:** `0.898 · ritmo 0.66`
* **Hartmann6:** `0.434 · ritmo 0.70`
* **Borehole:** `0.720 · ritmo 0.96`

### Nota pequeña junto al gráfico

**Ritmo alto = la mejora aparece pronto durante la trayectoria.**

La mejora acumulada se usa precisamente para distinguir si una configuración encuentra buenas soluciones pronto o si solo mejora al final del presupuesto disponible.

---

## Caja lateral: lectura

**Lectura**

* **Branin:** mayor mejora final.
* **Borehole:** mejora temprana y fuerte.
* **Forrester:** mejora alta, pero más tardía.
* **Hartmann6:** mejora moderada en mayor dimensión.

---

## Banda inferior

**Mejor MAE (\neq) mejor optimización**

El modelo que predice mejor en promedio no siempre es el que mejor guía la búsqueda del mínimo.

Esta idea está directamente respaldada por el TFG: en Hartmann6, por ejemplo, el mejor modelo según MAE es GP Linear, pero el mejor para mejorar el mínimo encontrado es GP RBF; en Branin y Forrester, GP RBF destaca en MAE, mientras que los mejores modelos de optimización son Matérn.

---

## Frase final de la slide

**En SBO no basta con reducir error medio: importa encontrar mejores valores con pocas evaluaciones.**

---

# Diapositiva 15 — Qué condiciona el rendimiento

## Mensaje de la diapositiva

**El rendimiento del proceso GP + EI no depende de un único factor: surge de la interacción entre benchmark, dimensión, diseño inicial y kernel.**

---

## Título

**Qué condiciona el rendimiento de GP + EI**

## Subtítulo

**La mejora depende del paisaje objetivo, del tamaño inicial y de la familia de kernel; no hay una configuración universalmente óptima.**

---

## Bloque 1 — Evidencia agregada

Este bloque debe ser visualmente fuerte.

### Texto grande

**316 / 372 trayectorias GP reducen el \emph{incumbent}**

### Texto pequeño debajo

Contraste complementario sobre trayectorias activas de \emph{infill}: reducción sistemática del mejor valor observado.

### Mini-tabla opcional

| Familia         | Trayectorias que mejoran |
| --------------- | -----------------------: |
| Matérn 5/2 ARD |                  54 / 54 |
| Matérn 5/2     |                  60 / 66 |
| RBF             |                  59 / 66 |
| RBF ARD         |                  51 / 54 |
| Lineal          |                  30 / 66 |

La lectura correcta es que el proceso de \emph{infill} reduce el \emph{incumbent} de forma sistemática en las trayectorias consideradas, con efecto especialmente claro en Matérn/RBF y sus variantes ARD.

---

## Bloque 2 — Factor dominante: benchmark y dimensión

**Benchmark y dimensión**

* La mejora cambia mucho entre funciones.
* Branin alcanza **0.898**.
* Hartmann6 queda en **0.434**.
* La geometría del problema condiciona más que una decisión aislada del pipeline.

Esto debe escribirse con prudencia: no es que Hartmann6 “falle”, sino que es el escenario más exigente por mayor dimensión y estructura del problema. En el TFG se indica que la dificultad del paisaje objetivo domina el resultado y que dimensión y estructura condicionan más que cualquier decisión aislada.

---

## Bloque 3 — Compromiso (n_}) vs \emph

**Diseño inicial**

[
n_{\text{train}}\in{1,d,4d}
]

* **Pocos puntos:** más margen de mejora relativa, pero más inestabilidad.
* **Más puntos:** GP más informado, pero menor margen relativo para que EI mejore.
* En mayor dimensión, el diseño inicial se vuelve más crítico.

Frase importante para evitar errores:

**No significa que menos puntos sea siempre mejor: el tamaño inicial condiciona la trayectoria y debe leerse junto al valor final alcanzado.**

Esto encaja con el TFG: el efecto del tamaño inicial es visible dentro de un mismo benchmark, pero con pocos puntos hay más margen de mejora relativa y también más inestabilidad.

---

## Bloque 4 — Consistencia de kernels

**Kernel**

* RBF y Matérn concentran los comportamientos más estables.
* El lineal puede destacar en casos concretos, pero no domina de forma general.
* ARD ayuda más en MAE que en optimización.

La frase defendible es:

**RBF y Matérn ofrecen el comportamiento más consistente, aunque la elección final depende de la geometría del benchmark y del criterio analizado.**

Esto está exactamente alineado con la lectura del TFG: el kernel importa, pero no debe presentarse como una competición con un único ganador; RBF y Matérn son más consistentes, mientras que ARD no mejora sistemáticamente la optimización aunque puede ayudar en calidad predictiva.

---

## Bloque 5 — Lectura prudente

**Lectura prudente**

* El sampler y el ruido modifican trayectorias concretas.
* No aparecen como factores dominantes frente al benchmark, (n_{\text{train}}) o kernel.
* El resultado no debe leerse como una regla universal.

---

## Frase final de la slide

**El éxito de SBO aparece cuando el diseño inicial, el kernel y EI encajan con la geometría del problema.**

---

# Texto oral recomendado

## Para la diapositiva 14

> “En esta primera diapositiva de resultados, la métrica principal no es el MAE, sino la mejora del \emph{incumbent}, es decir, el mejor valor encontrado durante el proceso. El resultado es positivo en los cuatro benchmarks. Branin tiene la mayor mejora final, Borehole mejora de forma fuerte y temprana, Forrester mejora bastante pero más tarde, y Hartmann6 es el caso más exigente por dimensionalidad.”

> “Por eso incluyo también el ritmo. No solo importa cuánto mejora el proceso al final, sino cuándo aparece esa mejora. En un contexto de presupuesto limitado, una mejora temprana puede ser especialmente valiosa.”

> “La lectura metodológica es que el mejor predictor medio no tiene por qué ser el mejor optimizador. En SBO no basta con reducir MAE: importa si el modelo y EI ayudan a encontrar mejores valores con pocas evaluaciones.”

## Para la diapositiva 15

> “La segunda lectura es agregada. No estoy mostrando solo trayectorias favorables: sobre 372 trayectorias GP, 316 reducen el \emph{incumbent}. Esto apoya que el proceso de \emph{infill} mejora de forma sistemática el mejor valor observado.”

> “Aun así, el rendimiento no depende de un único factor. El factor principal es el propio benchmark: su dimensión y su geometría. También importa el tamaño inicial. Con pocos puntos hay más margen de mejora relativa, pero también más inestabilidad; con más puntos el GP arranca mejor informado, pero el \emph{infill} tiene menos margen relativo para mejorar.”

> “Finalmente, el kernel importa, pero no hay un ganador universal. RBF y Matérn 5/2 son las familias más consistentes en conjunto, mientras que el kernel lineal puede destacar en casos concretos, como Borehole, pero no domina de forma general.”

---

# Prompt para el agente diseñador

Crea **dos diapositivas** para la defensa de mi TFG, manteniendo el estilo sobrio, editorial y académico de la presentación: fondo claro cálido, texto oscuro suave, azul petróleo para estructura, ámbar para elementos relacionados con EI/infill y verde para mejora o validación. No sobrecargar. Fórmulas en LaTeX donde aparezcan.

## Diapositiva 14

### Título

**Resultados sintéticos: mejora y ritmo de búsqueda**

### Subtítulo

**EI mejora el mejor valor encontrado en los cuatro benchmarks; la dificultad del problema condiciona tanto la mejora final como cuándo aparece.**

### Visual principal

Crear un gráfico de barras con la mejora final del \emph{incumbent}:

* Forrester: 0.737, ritmo 0.530
* Branin: 0.898, ritmo 0.663
* Hartmann6: 0.434, ritmo 0.701
* Borehole: 0.720, ritmo 0.963

La barra representa la mejora final. Añadir encima o cerca de cada barra una etiqueta tipo:

* `0.737 · ritmo 0.53`
* `0.898 · ritmo 0.66`
* `0.434 · ritmo 0.70`
* `0.720 · ritmo 0.96`

Añadir nota pequeña:
**Ritmo alto = la mejora aparece pronto durante la trayectoria.**

### Caja lateral

Título: **Lectura**

* **Branin:** mayor mejora final.
* **Borehole:** mejora temprana y fuerte.
* **Forrester:** mejora alta, pero más tardía.
* **Hartmann6:** mejora moderada en mayor dimensión.

### Banda inferior

Texto destacado:

**Mejor MAE (\neq) mejor optimización**

Texto secundario:

**En SBO no basta con reducir error medio: importa encontrar mejores valores con pocas evaluaciones.**

---

## Diapositiva 15

### Título

**Qué condiciona el rendimiento de GP + EI**

### Subtítulo

**La mejora depende del paisaje objetivo, del tamaño inicial y de la familia de kernel; no hay una configuración universalmente óptima.**

### Bloque fuerte de evidencia agregada

Mostrar grande:

**316 / 372 trayectorias GP reducen el \emph{incumbent}**

Texto pequeño:

**Contraste complementario sobre trayectorias activas de \emph{infill}.**

Añadir una mini-tabla o tarjetas pequeñas:

| Familia         | Trayectorias que mejoran |
| --------------- | -----------------------: |
| Matérn 5/2 ARD |                  54 / 54 |
| Matérn 5/2     |                  60 / 66 |
| RBF             |                  59 / 66 |
| RBF ARD         |                  51 / 54 |
| Lineal          |                  30 / 66 |

### Bloque de factores

Crear tres tarjetas:

**Benchmark y dimensión**

* Branin: **0.898**
* Hartmann6: **0.434**
* La geometría del problema domina el resultado.

**Diseño inicial**

[
n_{\text{train}}\in{1,d,4d}
]

* Pocos puntos: más margen, más inestabilidad.
* Más puntos: GP más informado, menor margen relativo.
* En mayor dimensión, el diseño inicial es más crítico.

**Kernel**

* RBF y Matérn son los más consistentes.
* El lineal destaca solo en casos concretos.
* ARD ayuda más en MAE que en optimización.

### Frase final

**El éxito de SBO aparece cuando el diseño inicial, el kernel y EI encajan con la geometría del problema.**

### Qué evitar

No convertir la diapositiva en una tabla densa. No presentar los contrastes como demostración de superioridad universal. No afirmar que menos datos iniciales siempre es mejor. No decir que RBF o Matérn ganan siempre.
