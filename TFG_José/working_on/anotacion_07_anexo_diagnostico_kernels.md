# Anotación 07: añadir lectura de kernels en el diagnóstico complementario

## Anotación o motivo

La anotación pide aprovechar mejor las figuras complementarias del anexo. La lectura que conviene añadir es que los kernels RBF y Matérn tienden a comportarse mejor de forma general, mientras que el kernel lineal es menos estable y solo destaca de forma puntual.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsubsection{Evolución de la capacidad predictiva del GP}`.

## Rango de sustitución

Sustituir:

```latex
Estos resultados confirman que la precisión global, la calibración probabilística y la utilidad para guiar la búsqueda del óptimo son aspectos relacionados, pero no equivalentes. Por ello, en este experimento se analizan por separado la mejora del mínimo encontrado y la evolución de la capacidad predictiva del modelo. Las figuras complementarias del Anexo~\ref{app:benchmarks_complementario} muestran con más detalle la evolución relativa del MAE y el diagnóstico conjunto entre MAE, NLPD y cobertura predictiva; véanse las Figuras~\ref{fig:app_evolution_relative_mae_best_model} y~\ref{fig:app_mae_vs_probabilistic_diagnostics}.
```

## Texto propuesto

```latex
Estos resultados confirman que la precisión global, la calibración probabilística y la utilidad para guiar la búsqueda del óptimo son aspectos relacionados, pero no equivalentes. Las figuras complementarias del Anexo~\ref{app:benchmarks_complementario} muestran con más detalle esta separación: RBF y Matérn concentran los comportamientos más estables en varios benchmarks, mientras que el kernel lineal resulta menos robusto y solo destaca en casos concretos; véanse las Figuras~\ref{fig:app_evolution_relative_mae_best_model} y~\ref{fig:app_mae_vs_probabilistic_diagnostics}.
```

## Relación con otras partes

- Esta idea conecta con el Cuadro de factores, donde el kernel aparece como factor relevante pero no universal.
- No conviene decir que el kernel lineal "no sirve": en algunos agregados aparece competitivo de forma puntual. La idea defendible es que es menos robusto.

## Notas fuera del LaTeX

La tabla de MAE agregado muestra a RBF y Matérn entre las mejores opciones en varios benchmarks, y al lineal con resultados malos en Branin y Forrester. En Hartmann6 el lineal aparece bien en MAE, por eso la redacción debe ser matizada.
