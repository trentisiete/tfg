# Anotación 19: aclarar que MAE macro es error medio por dieta

## Anotación o motivo

La profesora pregunta si debería decir "medio" en lugar de "macro". La métrica usada en la figura es efectivamente `mae_macro_mean`, por lo que el término técnico "macro" es correcto. El problema es de lectura: en el capítulo debe quedar claro que, en LODO, MAE macro significa error medio por dieta, porque se promedia el MAE de los folds externos y cada dieta pesa lo mismo.

La propuesta no elimina "macro"; lo acompaña con una formulación legible: "MAE medio por dieta (MAE macro)".

## Verificación realizada

- La figura `fig_04_kernel_family_mae.png` se genera con `mae_macro_mean` en `generate_entomotive_pov_figures.py`.
- El archivo `outputs/plots/TFG_MAIN_real_case_results_pov_ei_no_tpc/fig_04_kernel_family_mae_data.csv` contiene la columna `mae_macro_mean`.
- El GP compuesto es el mejor por `mae_macro_mean` en FCR, quitina y proteína dentro de esa figura.

## Ubicación

Archivo de referencia: `TFG_José/chapters/03_metodologia.tex`.

Secciones afectadas:

- `\subsection{Modelos, ajuste y métricas}`
- `\subsubsection{Expresividad geométrica del GP compuesto}`

## Cambio 1: separar y aclarar la explicación del MAE macro

Sustituir el párrafo largo que empieza por:

```latex
La lectura de métricas se apoya en las definiciones predictivas y probabilísticas introducidas en la Sección~\ref{metricas_estado_arte}
```

y termina en:

```latex
Esta distinción evita presentar una recomendación prospectiva como si fuera un resultado experimental confirmado.
```

por:

```latex
La lectura de métricas se apoya en las definiciones predictivas y probabilísticas introducidas en la Sección~\ref{metricas_estado_arte}, pero en este caso es necesario precisar cómo se agregan bajo validación LODO. El MAE se utiliza como métrica principal para seleccionar hiperparámetros y comparar modelos, porque mantiene las unidades originales de cada objetivo y permite interpretar directamente el error medio.

En LODO, cada fold externo corresponde a una dieta retenida. Por ello, el MAE macro se interpreta como el error medio por dieta: se calcula promediando el MAE de los folds externos, de modo que cada dieta tiene el mismo peso. Esta es la lectura principal del caso real, ya que el objetivo es evaluar la generalización a dietas completas no vistas, no favorecer dietas con más réplicas válidas.

Para comparar modelos frente a la línea base, se utiliza el MAE relativo respecto a Dummy:
\[
\rho_{\mathrm{Dummy}} = \frac{\mathrm{MAE}_{\mathrm{modelo}}}{\mathrm{MAE}_{\mathrm{Dummy}}}.
\]
Valores de \(\rho_{\mathrm{Dummy}}<1\) indican que el modelo reduce el error frente a la predicción trivial. Además, cuando se analiza la dificultad de cada dieta retenida, el error se normaliza por el rango del objetivo correspondiente. Esta normalización permite comparar visualmente errores de variables con escalas distintas, como FCR, quitina y proteína.

En los modelos GP también se evalúa la incertidumbre predictiva. La cobertura del intervalo aproximado del 95\,\% mide la proporción de observaciones que caen dentro de \(\mu \pm 1.96\sigma\). Además, se compara la desviación estándar predictiva \(\sigma(\bm{x})\) con el error absoluto observado \(|y-\mu(\bm{x})|\), con el fin de comprobar si el modelo asigna más incertidumbre a los casos donde efectivamente comete errores mayores. Estas métricas de incertidumbre no se utilizan para seleccionar parámetros, sino como diagnóstico de fiabilidad del GP.

Finalmente, en la parte prospectiva del caso real, EI se interpreta como una puntuación de adquisición y no como una métrica de validación. El \emph{incumbent} observado representa la mejor dieta ya medida experimentalmente para cada objetivo, mientras que el candidato con mayor EI es una formulación no ensayada que el modelo considera informativa por su combinación de media predictiva e incertidumbre. Esta distinción evita presentar una recomendación prospectiva como si fuera un resultado experimental confirmado.
```

## Cambio 2: aclarar el texto que introduce la figura de kernels

Sustituir:

```latex
La Figura~\ref{fig:kernel_family_mae} compara distintas familias de kernel utilizando el MAE macro en el protocolo LODO.
```

por:

```latex
La Figura~\ref{fig:kernel_family_mae} compara distintas familias de kernel utilizando el MAE medio por dieta, es decir, el MAE macro del protocolo LODO.
```

## Cambio 3: aclarar el pie de figura

Sustituir:

```latex
\caption[Comparación de kernels en Hermetia]{Comparación de familias de kernel en el caso real de \textit{Hermetia}. El eje vertical muestra el MAE macro obtenido mediante validación LODO; valores menores indican mejor generalización a dietas no vistas.}
```

por:

```latex
\caption[Comparación de kernels en Hermetia]{Comparación de familias de kernel en el caso real de \textit{Hermetia}. El eje vertical muestra el MAE medio por dieta, equivalente al MAE macro obtenido mediante validación LODO; valores menores indican mejor generalización a dietas no vistas.}
```

## Cambio opcional si se regenera la figura

Si se vuelve a generar `fig_04_kernel_family_mae.png`, conviene cambiar la etiqueta del eje en `generate_entomotive_pov_figures.py`:

```python
g.set_axis_labels("", "MAE medio por dieta (macro LODO)")
```

## Relación con otras partes

- El término técnico se conserva porque coincide con la métrica calculada.
- La explicación se hace más accesible para quien lee la memoria sin detenerse en nomenclatura técnica.
- La figura queda conectada con la definición previa de LODO: cada fold es una dieta retenida.

## Notas fuera del LaTeX

No usar solo "MAE medio" sin más, porque podría confundirse con un promedio por observación. En este caso lo importante es que el promedio es por dieta.

Se elimina la explicación de MAE micro porque no se utiliza en los resultados del capítulo. Mencionarlo añade una distinción que no se aprovecha después.
