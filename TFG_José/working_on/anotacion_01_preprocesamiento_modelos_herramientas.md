# Anotación 01: separar protocolo experimental de herramienta de implementación

## Anotación o motivo

La anotación pide separar con claridad:

- los pasos experimentales replicables: preprocesamiento concreto, tipo de escalado y modelos utilizados;
- la herramienta o implementación usada: clases propias, `Pipeline`, lenguaje, paquetes y entorno de ejecución.

También pide definir mejor el modelo `Dummy`, porque no siempre tiene por qué significar lo mismo.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\section{Marco común de modelado sustituto}`, dentro de `\subsection{Datos, modelos y preprocesamiento}`.

## Rango de sustitución 1

Sustituir desde:

```latex
El diseño experimental considera dos modelos. El modelo Dummy se utiliza como línea base mínima para comprobar si el GP aporta capacidad predictiva frente a una predicción trivial. El proceso gaussiano constituye el modelo sustituto principal y se implementa mediante la clase \texttt{GPSurrogateRegressor}, construida a partir de un \texttt{Pipeline} que integra el escalado de variables y el ajuste de un \texttt{GaussianProcessRegressor}.
```

hasta:

```latex
El preprocesamiento se realiza siempre dentro del \texttt{Pipeline}, de modo que el escalado se ajusta únicamente con los datos de entrenamiento de cada partición o iteración experimental. Esto evita utilizar información del conjunto de evaluación durante el ajuste. En el caso real, las transformaciones adicionales necesarias para representar las dietas se describen en la sección \ref{subsec:experimento_caso_real}.
```

## Texto propuesto 1

```latex
El diseño experimental compara una línea base constante y varias configuraciones de GP. Antes del ajuste, las variables de entrada se estandarizan mediante \emph{z-score}: cada columna se centra con la media del conjunto de entrenamiento y se divide por su desviación típica. Esta transformación se estima de nuevo dentro de cada partición o iteración experimental y después se aplica al conjunto de evaluación correspondiente, evitando así utilizar información de test durante el ajuste. En el caso real, las transformaciones adicionales necesarias para representar las dietas se describen en la Sección~\ref{subsec:experimento_caso_real}.

El modelo Dummy actúa como referencia mínima. En cada entrenamiento predice un valor constante calculado a partir de las respuestas del conjunto de entrenamiento: en los benchmarks se utiliza la media, mientras que en el caso real se considera media o mediana dentro de la selección interna. Su función no es modelar la relación entre \(X\) e \(\bm{y}\), sino comprobar si el GP aporta información por encima de una predicción que ignora las covariables.

El modelo sustituto principal es un GP. En los benchmarks se comparan kernels lineales, RBF y Matérn, incluyendo variantes con ARD cuando la dimensión lo permite. En el caso real se evalúan esas mismas familias junto con un kernel compuesto que combina tendencia global, variación local y ruido experimental. Esta descripción define el protocolo experimental; las clases y paquetes concretos empleados para ejecutarlo se describen por separado.
```

## Rango de inserción 2

Añadir una nueva subsección común después del párrafo de `\subsection{Predicción probabilística e incertidumbre}` y antes de:

```latex
\section{Experimento 1: benchmarks sintéticos}
```

## Texto propuesto 2

```latex
\subsection{Entorno de implementación}
\label{subsubsec:entorno_implementacion}

Los experimentos se implementaron en Python, usando \texttt{pandas} y \texttt{numpy} para la manipulación de datos, \texttt{scikit-learn} para el ajuste y validación de modelos, \texttt{scipy} y \texttt{scikit-optimize} para utilidades numéricas y optimización, y \texttt{matplotlib}, \texttt{seaborn} y \texttt{plotly} para la generación de figuras. Las ejecuciones se realizaron en un equipo con Windows 10.0.26200 de 64 bits, procesador Intel(R) Core(TM) Ultra 7 155H, 22 hilos lógicos y 31.6 GB de RAM.
```

## Relación con otras partes

- La Sección `\ref{subsec:preprocesamiento_caso_real}` ya detalla los pasos específicos de limpieza y representación del caso real. Por eso aquí conviene explicar solo el preprocesamiento común y remitir allí los detalles.
- La Sección `\ref{subsec:modelos_metricas_caso_real}` contiene la tabla de modelos del caso real. El texto propuesto evita duplicarla, pero deja definido el papel del Dummy y del GP.
- Las métricas posteriores usan Dummy como referencia de comparación. Definirlo aquí evita ambigüedad cuando se interpreta el MAE relativo frente a Dummy.

## Notas

El dato de sistema se obtuvo por terminal. No incluir dudas o pendientes dentro del bloque LaTeX.
