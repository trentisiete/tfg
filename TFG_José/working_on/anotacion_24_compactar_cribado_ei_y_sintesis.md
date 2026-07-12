# Anotación 24: compactar cribado EI y síntesis final

## Anotación o motivo

La idea de que EI no valida nuevas dietas aparece en varios puntos: métricas del caso real, entrada del cribado, frase previa al cuadro, párrafo posterior al cuadro, cierre de la subsección y síntesis final. Es importante, pero se puede concentrar sin perder claridad.

## Prioridad

Media. Ahorra espacio al final del capítulo, donde el texto ya está muy cargado de cierres.

## Ubicación

Archivo: `TFG_José/chapters/03_metodologia.tex`.

Sección: `\subsubsection{Cribado prospectivo mediante EI}` y `\subsection{Síntesis de resultados}`.

## Cambio 1: compactar la introducción de EI

Sustituir:

```latex
Como cierre del caso real, se analiza si el GP entrenado puede utilizarse como herramienta de cribado para una posible fase futura de experimentación. A diferencia del experimento de benchmarks, aquí no se ejecuta un ciclo físico de \emph{infill}: no se ensayan nuevas dietas ni se actualiza el conjunto de datos. El objetivo es únicamente comprobar si el modelo permite priorizar formulaciones candidatas dentro de un espacio factible.
```

por:

```latex
Como cierre del caso real, se analiza si el GP entrenado puede servir como herramienta de cribado para una fase experimental futura. Aquí no se ejecuta un ciclo físico de \emph{infill}: solo se priorizan formulaciones candidatas dentro de un espacio factible.
```

## Cambio 2: compactar la frase previa al cuadro

Sustituir:

```latex
La Cuadro~\ref{tab:incumbents_candidatos_ei} compara, para cada objetivo, el mejor punto observado experimentalmente con el candidato no ensayado priorizado por EI. Esta distinción es importante: el \emph{incumbent} observado es una evidencia experimental, mientras que el candidato EI es una hipótesis propuesta por el modelo para una posible evaluación posterior.
```

por:

```latex
El Cuadro~\ref{tab:incumbents_candidatos_ei} compara, para cada objetivo, el mejor punto observado experimentalmente con el candidato no ensayado priorizado por EI.
```

## Cambio 3: compactar el cierre de EI

Sustituir:

```latex
Los resultados muestran que los candidatos EI no sustituyen a los mejores valores observados: para FCR el incumbent sigue siendo \texttt{Quinoa30}, para quitina \texttt{Orujo70} y para proteína \texttt{Orujo50}. Además, las medias predichas de los candidatos no superan claramente a dichos incumbents. Por tanto, la utilidad de EI en este caso no reside en demostrar nuevas dietas óptimas, sino en transformar la predicción media y la incertidumbre del GP en propuestas concretas para una siguiente iteración experimental.

En conjunto, este análisis debe interpretarse como una fase de pre-\emph{infill}. El modelo sustituto reduce el espacio de búsqueda y señala formulaciones con cierto valor informativo, pero la decisión final requeriría validación experimental. Una visualización complementaria del paisaje de EI sobre la rejilla factible se recoge en el Anexo~\ref{app:ei_landscape_caso_real}.
```

por:

```latex
Los candidatos EI no sustituyen a los mejores valores observados: para FCR el \emph{incumbent} sigue siendo \texttt{Quinoa30}, para quitina \texttt{Orujo70} y para proteína \texttt{Orujo50}. Su utilidad es convertir la media predictiva y la incertidumbre del GP en propuestas concretas para una siguiente iteración experimental, que requeriría validación física. Una visualización complementaria del paisaje de EI sobre la rejilla factible se recoge en el Anexo~\ref{app:ei_landscape_caso_real}.
```

## Cambio 4: compactar la síntesis final

Sustituir:

```latex
En conjunto, los resultados del capítulo muestran una lectura coherente entre los dos escenarios experimentales. En los benchmarks sintéticos, el ciclo GP + EI permite mejorar el mejor valor encontrado respecto al diseño inicial, especialmente cuando la incertidumbre del modelo ayuda a dirigir nuevas evaluaciones. En el caso real, la validación LODO indica que el GP compuesto aporta capacidad predictiva frente al modelo Dummy en varios objetivos, aunque su incertidumbre no aparece perfectamente calibrada en todos ellos. Por tanto, el modelo sustituto no debe interpretarse como una herramienta capaz de reemplazar la validación experimental, sino como un mecanismo para reducir el espacio de búsqueda, cuantificar parcialmente la incertidumbre y proponer candidatos razonables para una siguiente iteración de ensayos.
```

por:

```latex
En conjunto, los resultados muestran una lectura coherente entre escenarios. En benchmarks, el ciclo GP + EI mejora el mejor valor encontrado respecto al diseño inicial. En el caso real, la validación LODO indica que el GP compuesto aporta capacidad predictiva frente a Dummy, aunque con incertidumbre solo parcialmente calibrada. Por tanto, el modelo sustituto no reemplaza la validación experimental, pero sí ayuda a reducir el espacio de búsqueda y a proponer candidatos razonables para ensayos posteriores.
```

## Ahorro estimado

5--8 líneas.

## Relación con otras partes

- Se mantiene la cautela principal sobre EI.
- Se corrige `La Cuadro` por `El Cuadro`.
- La síntesis final queda más breve y no repite todos los matices ya desarrollados en las subsecciones anteriores.

## Notas fuera del LaTeX

Este recorte es seguro si las subsecciones anteriores conservan las cifras y explicaciones principales.
