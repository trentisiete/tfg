# Anotación 03: referenciar tablas y figuras en el texto

## Anotación o motivo

La profesora indica que toda tabla y figura debe aparecer referenciada en el texto, dejando claro qué aporta y por qué es importante. Se revisaron las tablas y figuras de `TFG_José/working_on/cap3.tex`.

## Diagnóstico de referencias

Elementos sin referencia textual explícita:

- `tab:benchmarks_seleccionados`
- `tab:configuracion_benchmarks`
- `tab:dataset_caso_real`
- `tab:modelos_caso_real`

Elementos ya referenciados al menos una vez:

- `tab:resumen_incumbent_benchmarks`
- `fig:evolution_incumbent_best_model`
- `tab:resumen_predictivo_benchmarks`
- `tab:jerarquia_factores_benchmark`
- `fig:nested_lodo`
- `fig:lodo_generalization_vs_dummy`
- `fig:error_by_left_out_diet`
- `fig:kernel_family_mae`
- `fig:uncertainty_vs_error`
- `tab:incumbents_candidatos_ei`

## Cambio 1: tabla de benchmarks seleccionados

### Ubicación

Antes de `\begin{table}[htbp]` de `tab:benchmarks_seleccionados`, dentro de `\subsection{Benchmarks seleccionados}`.

### Rango de sustitución

Sustituir:

```latex
Se consideran cuatro benchmarks con una progresión razonable de dificultad, desde problemas de baja dimensión e interpretación visual hasta funciones de mayor dimensión que deben analizarse principalmente mediante métricas. Los dominios de búsqueda se definen en la implementación y se utilizan de forma común para el muestreo inicial, el conjunto de test y la optimización del criterio de adquisición.
```

por:

```latex
Se consideran cuatro benchmarks con una progresión razonable de dificultad, desde problemas de baja dimensión e interpretación visual hasta funciones de mayor dimensión que deben analizarse principalmente mediante métricas. Los dominios de búsqueda se definen en la implementación y se utilizan de forma común para el muestreo inicial, el conjunto de test y la optimización del criterio de adquisición. Esta selección se resume en el Cuadro~\ref{tab:benchmarks_seleccionados}, que muestra la dimensión, el dominio y el papel metodológico de cada función dentro del experimento.
```

## Cambio 2: tabla de configuración de benchmarks

### Ubicación

Antes de `\begin{table}[htbp]` de `tab:configuracion_benchmarks`, dentro de `\subsection{Diseño inicial, presupuesto y configuraciones GP}`.

### Rango de sustitución

Sustituir desde:

```latex
A partir del diseño inicial, el proceso de \emph{infill} añade \(5 \times d\) nuevos puntos, sujeto a un máximo de 50 observaciones durante la trayectoria. También se consideran tres condiciones de ruido: evaluación sin ruido y ruido gaussiano con niveles \(0.5\) y \(1.0\).
```

hasta:

```latex
El dominio detallado de Borehole, el flujo completo del proceso de \emph{infill} y la configuración específica de semillas se incluyen como material complementario en el Anexo~\ref{app:benchmarks_complementario}. En particular, el dominio de Borehole se recoge en el Cuadro~\ref{tab:app_dominio_borehole}, el esquema del proceso secuencial en la Figura~\ref{fig:app_flujo_infill} y las semillas empleadas en el Cuadro~\ref{tab:app_semillas_benchmarks}.
```

por:

```latex
A partir del diseño inicial, el proceso de \emph{infill} añade \(5 \times d\) nuevos puntos, sujeto a un máximo de 50 observaciones durante la trayectoria. También se consideran tres condiciones de ruido: evaluación sin ruido y ruido gaussiano con niveles \(0.5\) y \(1.0\). La configuración general se resume en el Cuadro~\ref{tab:configuracion_benchmarks}, que fija las condiciones comunes empleadas después para agregar y comparar los resultados.

El dominio detallado de Borehole, el flujo completo del proceso de \emph{infill} y la configuración específica de semillas se incluyen como material complementario en el Anexo~\ref{app:benchmarks_complementario}. En particular, el dominio de Borehole se recoge en el Cuadro~\ref{tab:app_dominio_borehole}, el esquema del proceso secuencial en la Figura~\ref{fig:app_flujo_infill} y las semillas empleadas en el Cuadro~\ref{tab:app_semillas_benchmarks}.
```

## Cambio 3: tabla del dataset del caso real

### Ubicación

Antes de `\begin{table}[htbp]` de `tab:dataset_caso_real`, dentro de `\subsection{Diseño experimental y dietas consideradas}`.

### Rango de sustitución

Sustituir:

```latex
Cada dieta se administra a tres lotes de larvas, por lo que el análisis se realiza a nivel de réplica experimental. Esta decisión conserva la variabilidad entre lotes y evita reducir prematuramente el dataset a medias por dieta. La columna \texttt{diet\_name} identifica la dieta asociada a cada réplica y se utiliza posteriormente como variable de agrupación en la validación.
```

por:

```latex
Cada dieta se administra a tres lotes de larvas, por lo que el análisis se realiza a nivel de réplica experimental. Esta decisión conserva la variabilidad entre lotes y evita reducir prematuramente el dataset a medias por dieta. La columna \texttt{diet\_name} identifica la dieta asociada a cada réplica y se utiliza posteriormente como variable de agrupación en la validación. El Cuadro~\ref{tab:dataset_caso_real} resume estos elementos y fija el alcance del caso real antes de describir la representación de entrada.
```

## Cambio 4: tabla de modelos del caso real

### Ubicación

Antes de `\begin{table}[htbp]` de `tab:modelos_caso_real`, dentro de `\subsection{Modelos, ajuste y métricas}`.

### Rango de sustitución

Sustituir:

```latex
En esta etapa se comparan una línea base simple y varias configuraciones de GP. Todos los modelos se evalúan bajo el protocolo LODO anidado descrito en la Sección~\ref{subsec:lodo_caso_real}, de forma independiente para cada objetivo y para cada modo de representación de entrada: \texttt{REDUCED\_FEATURES} y \texttt{FULL\_FEATURES}.
```

por:

```latex
En esta etapa se comparan una línea base simple y varias configuraciones de GP. Todos los modelos se evalúan bajo el protocolo LODO anidado descrito en la Sección~\ref{subsec:lodo_caso_real}, de forma independiente para cada objetivo y para cada modo de representación de entrada: \texttt{REDUCED\_FEATURES} y \texttt{FULL\_FEATURES}. El Cuadro~\ref{tab:modelos_caso_real} resume las familias comparadas y explicita el papel de cada una dentro de este protocolo.
```

## Relación con otras partes

- Si la plantilla presenta las tablas como "Cuadro" en el documento final, las referencias textuales deben usar también "Cuadro" para mantener coherencia.
- Las figuras de resultados del caso real ya están referenciadas y explicadas en el texto, por lo que no se proponen cambios adicionales ahí.
- La revisión debe repetirse cada vez que se añada una nueva tabla o figura: no basta con incluir `\caption` y `\label`.

## Notas

Estas propuestas añaden referencias breves y funcionales, sin alargar el capítulo ni duplicar lo que ya explican las leyendas.
