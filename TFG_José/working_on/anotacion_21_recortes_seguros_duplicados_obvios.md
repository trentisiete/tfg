# Anotación 21: recortes seguros por duplicados obvios

## Anotación o motivo

Hay varios duplicados literales o casi literales que ocupan espacio sin aportar comprensión. Son cambios de bajo riesgo porque no eliminan contenido conceptual, solo repeticiones o referencias duplicadas.

## Prioridad

Alta. Estos cambios deberían aplicarse antes que cualquier recorte interpretativo.

## Cambio 1: fusionar la explicación de diseño, escalado y tabla de modelos

### Ubicación

Archivo: `TFG_José/chapters/03_metodologia.tex`.

Sección: `\subsection{Datos, modelos y preprocesamiento}`.

### Rango de sustitución

Sustituir desde:

```latex
El diseño experimental compara una línea base constante y varias configuraciones de GP.
```

hasta:

```latex
Esta descripción define el protocolo experimental; las clases y paquetes concretos empleados para ejecutarlo se describen por separado.
```

por:

```latex
El diseño experimental compara una línea base constante y varias familias de GP, resumidas en el Cuadro~\ref{tab:modelos_capitulo}. Antes del ajuste, las variables de entrada se estandarizan mediante \emph{z-score}: cada columna se centra con la media del conjunto de entrenamiento y se divide por su desviación típica. Esta transformación se estima de nuevo dentro de cada partición o iteración experimental y después se aplica al conjunto de evaluación correspondiente, evitando así utilizar información de test durante el ajuste. En el caso real, las transformaciones adicionales necesarias para representar las dietas se describen en la Sección~\ref{subsec:experimento_caso_real}.
```

### Ahorro estimado

1--2 líneas y menos repetición de "diseño compara".

## Cambio 2: dejar un único entorno de implementación

### Ubicación

Sección: `\subsection{Entorno de implementación}`.

### Rango de sustitución

Sustituir las dos subsecciones consecutivas `Entorno de implementación` por una sola:

```latex
\subsection{Entorno de implementación}
\label{subsubsec:entorno_implementacion}

Los experimentos se implementaron en Python. La manipulación de datos se realizó con \texttt{pandas} y \texttt{numpy}; el ajuste de modelos y los protocolos de validación se apoyaron en \texttt{scikit-learn}; la optimización de funciones de adquisición y las utilidades numéricas emplearon \texttt{scipy} y \texttt{scikit-optimize}; y las figuras se generaron con \texttt{matplotlib}, \texttt{seaborn} y \texttt{plotly}. En la implementación, los modelos se encapsularon en las clases \texttt{DummySurrogateRegressor} y \texttt{GPSurrogateRegressor}, esta última construida sobre un \texttt{Pipeline} con \texttt{StandardScaler} y \texttt{GaussianProcessRegressor}. Las ejecuciones se realizaron en un equipo con Windows 10.0.26200 de 64 bits, procesador Intel(R) Core(TM) Ultra 7 155H, 22 hilos lógicos y 31.6 GB de RAM.
```

### Ahorro estimado

4--6 líneas y elimina una etiqueta duplicada.

## Cambio 3: eliminar duplicado literal en métricas benchmark

### Ubicación

Sección: `\subsection{Métricas del experimento benchmark}`.

### Rango de sustitución

Sustituir:

```latex
La capacidad predictiva se analiza mediante el MAE relativo respecto al estado inicial y mediante la comparación frente al modelo Dummy. El MAE relativo permite comprobar si las observaciones añadidas reducen el error del GP sobre un conjunto de test independiente. La capacidad predictiva se analiza mediante el MAE relativo respecto al estado inicial y mediante la comparación frente al modelo Dummy. El MAE relativo permite comprobar si las observaciones añadidas reducen el error del GP sobre un conjunto de test independiente.
```

por:

```latex
La capacidad predictiva se analiza mediante el MAE relativo respecto al estado inicial y mediante la comparación frente al modelo Dummy. El MAE relativo permite comprobar si las observaciones añadidas reducen el error del GP sobre un conjunto de test independiente.
```

### Ahorro estimado

2 líneas.

## Cambio 4: corregir referencia duplicada al cuadro del dataset

### Ubicación

Sección: `\subsection{Diseño experimental y dietas consideradas}`.

### Rango de sustitución

Sustituir:

```latex
Cada formulación se ensayó en tres lotes independientes de larvas. Por ello, cada observación representa una réplica de una dieta, no una media por dieta. En la validación, las tres réplicas de una misma formulación se mantienen juntas: cuando una dieta se deja fuera, ninguna de sus réplicas participa en el entrenamiento. El Cuadro~\ref{tab:dataset_caso_real} resume el alcance del caso real.La Tabla~\ref{tab:dataset_caso_real} resume estos elementos y fija el alcance del caso real antes de describir la representación de entrada.
```

por:

```latex
Cada formulación se ensayó en tres lotes independientes de larvas. Por ello, cada observación representa una réplica de una dieta, no una media por dieta. En la validación, las tres réplicas de una misma formulación se mantienen juntas. El Cuadro~\ref{tab:dataset_caso_real} resume estos elementos y fija el alcance del caso real antes de describir la representación de entrada.
```

### Ahorro estimado

1--2 líneas y corrige `caso real.La Tabla`.

## Relación con otras partes

Estos cambios no alteran el razonamiento del capítulo. Solo eliminan repeticiones o duplicados que ya estaban generando ruido.

## Notas fuera del LaTeX

No afectan a los pasos de limpieza y preprocesamiento ya validados.
