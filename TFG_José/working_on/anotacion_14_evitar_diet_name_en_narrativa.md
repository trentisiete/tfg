# Anotación 14: evitar nombres internos como diet_name

## Anotación o motivo

`diet_name` es útil en la implementación, pero en la memoria conviene hablar de dieta, formulación o grupo de réplicas. El lector no necesita conocer el nombre exacto de la columna salvo en una sección estrictamente técnica.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsection{Diseño experimental y dietas consideradas}`.

## Rango de sustitución mínimo

Si no se aplica la anotación 13 completa, sustituir:

```latex
La columna \texttt{diet\_name} identifica la dieta asociada a cada réplica y se utiliza posteriormente como variable de agrupación en la validación.
```

## Texto propuesto mínimo

```latex
La agrupación se realiza por dieta común y se utiliza posteriormente en la validación.
```

## Relación con otras partes

- Si se aplica el texto completo de la anotación 13, este cambio ya queda incorporado y no hace falta aplicarlo por separado.
- En la subsección de limpieza puede seguir mencionándose la construcción de grupos, pero también conviene evitar el nombre interno de la columna cuando no sea imprescindible.

## Notas fuera del LaTeX

La opción recomendada es aplicar la anotación 13, porque explica mejor el problema sin alargar demasiado.
