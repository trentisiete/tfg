# Anotación 25: compactar la lectura temporal de mejora acumulada

## Anotación o motivo

En la subsección de evolución del mínimo encontrado se explica la mejora acumulada en la sección de métricas, después se vuelve a explicar antes del cuadro, y luego se interpreta dos veces la lectura temporal: primero en un párrafo por benchmark y después al comentar la figura. Se puede conservar la idea con menos texto.

## Prioridad

Media-baja. Aplicar si todavía hace falta espacio después de los recortes de prioridad alta.

## Ubicación

Archivo: `TFG_José/chapters/03_metodologia.tex`.

Sección: `\subsubsection{Evolución del mínimo encontrado durante el proceso de infill}`.

## Cambio 1: acortar la explicación de mejora acumulada

Sustituir:

```latex
La mejora acumulada se calcula a partir del área bajo la trayectoria normalizada de mejora relativa. Por tanto, no solo mide el resultado final, sino también la rapidez con la que aparece la mejora. Un ratio cercano a uno indica que la mayor parte de la mejora se obtiene pronto, mientras que un ratio menor sugiere una mejora más tardía.
```

por:

```latex
La mejora acumulada resume la trayectoria: ratios cercanos a uno indican mejoras tempranas, mientras que valores menores sugieren progreso más tardío.
```

## Cambio 2: fusionar los dos párrafos de lectura temporal

Sustituir desde:

```latex
En Borehole, la mejora final y la mejora acumulada son muy próximas,
```

hasta:

```latex
Como material complementario, la Figura~\ref{fig:app_final_relative_incumbent_improvement} muestra la mejora relativa final del mínimo encontrado agregada por benchmark.
```

por:

```latex
La Figura~\ref{fig:evolution_incumbent_best_model} confirma esta lectura temporal. En Borehole, la mejora aparece pronto; en Branin, el resultado final es muy alto aunque el progreso continúa durante la trayectoria; en Forrester, la mejora es fuerte pero menos inmediata; y en Hartmann6, el avance es más gradual, coherente con su mayor dimensionalidad. Como material complementario, la Figura~\ref{fig:app_final_relative_incumbent_improvement} muestra la mejora relativa final agregada por benchmark.
```

## Ahorro estimado

3--5 líneas.

## Relación con otras partes

- La definición formal de mejora relativa se mantiene en métricas.
- El cuadro conserva las columnas de mejora final, mejora acumulada y ratio.
- La figura sigue teniendo una interpretación clara, pero sin repetir dos veces la misma comparación por benchmark.

## Notas fuera del LaTeX

Este recorte es menos urgente que eliminar duplicados literales, pero útil si todavía falta espacio.
