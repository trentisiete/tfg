# Anotación 16: describir variables de entrada sin nombres internos

## Anotación o motivo

La redacción actual enumera nombres de columnas (`REDUCED_FEATURES`, `FULL_FEATURES`, `inclusion_pct`, `byproduct_type`). Eso es útil para programar, pero menos claro para quien lee la memoria sin ver los datos. Conviene describir qué información entra al modelo.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsection{Diseño experimental y dietas consideradas}`.

## Rango de sustitución principal

Sustituir:

```latex
La representación de entrada combina variables de formulación y composición de la dieta. Se consideran dos conjuntos de variables. El modo reducido, \texttt{REDUCED\_FEATURES}, incluye \texttt{inclusion\_pct}, \texttt{Proteína (\%)\_media}, \texttt{Fibra (\%)\_media}, \texttt{Grasa (\%)\_media} y \texttt{TPC\_dieta\_media}. El modo completo, \texttt{FULL\_FEATURES}, añade \texttt{Cenizas (\%)\_media}, \texttt{Carbohidratos (\%)\_media} y tres ratios nutricionales derivados. En ambos casos se incorporan variables binarias asociadas al tipo de subproducto mediante la codificación de \texttt{byproduct\_type}.
```

## Texto propuesto principal

```latex
La representación de entrada describe cada dieta mediante variables de formulación y composición. Se comparan dos conjuntos: uno reducido, con porcentaje de inclusión, proteína, fibra, grasa y TPC medios de la dieta; y otro completo, que añade cenizas, carbohidratos y ratios nutricionales derivados. En ambos casos se incluye el tipo de subproducto, de forma que el modelo distingue entre control, hoja de olivo, orujo de oliva y quinoa sin depender solo del nombre de la dieta.
```

## Cambios relacionados en limpieza y preprocesamiento

No aplicar estos cambios por ahora. La parte de pasos de limpieza y preprocesamiento ya fue valorada positivamente por la profesora, así que conviene conservarla salvo que aparezca una anotación nueva y explícita sobre esos pasos.

Se dejan aquí solo como referencia de relación con otras partes, no como sustituciones recomendadas.

### Paso 4

Referencia afectada:

```latex
A partir del nombre textual de cada dieta se generan variables estructuradas, como \texttt{diet\_name}, \texttt{byproduct\_type} e \texttt{inclusion\_pct}. Esta transformación permite que el modelo no dependa únicamente del nombre de la dieta, sino de variables interpretables: tipo de subproducto, porcentaje de inclusión y composición nutricional. Además, se calculan ratios nutricionales derivados que se emplean en el modo completo de variables.
```

### Paso 6

Referencia afectada:

```latex
Para cada objetivo se construye la matriz de entrada \(X\), el vector de respuesta \(\bm{y}\) y el vector de grupos. El vector \(\bm{y}\) contiene los valores observados del objetivo seleccionado. El vector de grupos se construye a partir de \texttt{diet\_name}, de forma que todas las réplicas de una misma dieta compartan el mismo identificador de grupo. Esta agrupación será la base del protocolo LODO descrito en la Sección~\ref{subsec:lodo_caso_real}.
```

### Paso 7

Referencia afectada:

```latex
Las variables de entrada se forman combinando las variables numéricas seleccionadas con una codificación binaria del tipo de subproducto. En concreto, \texttt{byproduct\_type} se transforma en variables indicadoras y se concatena con las variables numéricas del modo reducido o completo. Todas las columnas resultantes se convierten a formato numérico antes del ajuste del modelo.
```

## Relación con otras partes

- Esta propuesta hace más legible la descripción de los datos sin eliminar la reproducibilidad: la lógica sigue siendo algorítmica, pero no depende de nombres de columnas.
- Encaja con las anotaciones 13 y 14, que sustituyen `diet_name` por agrupación por dieta.
- Los pasos de limpieza y preprocesamiento se mantienen como están, porque ya tienen validación positiva.

## Notas fuera del LaTeX

Se evita `REDUCED_FEATURES`, `FULL_FEATURES`, `inclusion_pct` y `byproduct_type` en la narración principal. No extender este criterio a los pasos de limpieza y preprocesamiento mientras esa parte siga validada.
