# Anotación 13: explicar las réplicas como dietas agrupadas

## Anotación o motivo

El párrafo actual puede ser difícil para un lector que no ha visto los datos. La idea principal es más simple: cada dieta tiene tres réplicas y esas réplicas se mantienen juntas en la validación.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsection{Diseño experimental y dietas consideradas}`.

## Rango de sustitución

Sustituir:

```latex
Cada dieta se administra a tres lotes de larvas, por lo que el análisis se realiza a nivel de réplica experimental. Esta decisión conserva la variabilidad entre lotes y evita reducir prematuramente el dataset a medias por dieta. La columna \texttt{diet\_name} identifica la dieta asociada a cada réplica y se utiliza posteriormente como variable de agrupación en la validación.
```

## Texto propuesto

```latex
Cada formulación se ensayó en tres lotes independientes de larvas. Por ello, cada observación representa una réplica de una dieta, no una media por dieta. En la validación, las tres réplicas de una misma formulación se mantienen juntas: cuando una dieta se deja fuera, ninguna de sus réplicas participa en el entrenamiento. El Cuadro~\ref{tab:dataset_caso_real} resume el alcance del caso real.
```

## Relación con otras partes

- Esta redacción anticipa el protocolo LODO sin introducir todavía todo el detalle técnico.
- También resuelve la anotación sobre `diet_name` si se aplica este cambio completo.

## Notas fuera del LaTeX

Sí: decir que se agrupan por dieta común es la idea clave, pero conviene añadir una frase sobre qué implica en validación para que se entienda por qué importa.
