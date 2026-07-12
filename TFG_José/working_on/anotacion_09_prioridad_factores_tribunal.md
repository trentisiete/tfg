# Anotación 09: aclarar en qué centrarse tras el cuadro de factores

## Anotación o motivo

La anotación pide que el texto deje una lectura práctica del cuadro: si hay muchos factores, debe quedar claro cuáles son centrales y cuáles son secundarios. Esto también prepara una respuesta defendible ante una pregunta del tribunal.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsubsection{Influencia de las condiciones experimentales}`, después del cuadro `tab:jerarquia_factores_benchmark`.

## Rango de sustitución

Sustituir:

```latex
La conclusión principal es que el rendimiento del ciclo GP + EI está condicionado ante todo por el propio benchmark. Las diferencias entre funciones son mayores que
las asociadas a factores individuales como ARD o sampler. Esto indica que la dificultad geométrica del problema, la dimensionalidad y el margen de mejora disponible
determinan el comportamiento global del proceso.
El análisis gráfico de los efectos emparejados de estas condiciones se recoge como material complementario en la Figura~\ref{fig:app_factor_effects_paired}.
```

## Texto propuesto

```latex
La conclusión principal es que el rendimiento del ciclo GP + EI está condicionado ante todo por el propio benchmark. En una lectura práctica, el foco debe ponerse primero en la dificultad geométrica del problema, el tamaño del diseño inicial y la familia de kernel; ARD, el sampler y el ruido se interpretan como factores secundarios, capaces de modificar trayectorias concretas pero no de compensar por sí solos un problema mal condicionado o con poca información inicial. El análisis gráfico de los efectos emparejados de estas condiciones se recoge como material complementario en la Figura~\ref{fig:app_factor_effects_paired}.
```

## Relación con otras partes

- Refuerza el objetivo del cuadro: no listar factores, sino orientar la interpretación.
- Conecta con la discusión de resultados: optimizar bien no depende de un único ajuste técnico, sino del problema y de la información inicial disponible.

## Notas fuera del LaTeX

Respuesta breve si preguntan en tribunal: me centraría en tres decisiones principales, el tipo de problema, el tamaño del diseño inicial y la familia de kernel. ARD, sampler y ruido son importantes para analizar sensibilidad, pero no aparecen como palancas principales en estos resultados.
