# Anotación 17: ubicar los modelos en el marco común sin mezclar protocolos

## Anotación o motivo

La tabla de modelos del caso real no debería quedar como una explicación aislada si parte de esas familias también se usan en los benchmarks. Sin embargo, no todas las familias aparecen en ambos experimentos: el kernel compuesto se usa en el caso real, pero no en benchmarks.

La solución recomendada es mover la explicación general al marco común, pero con una columna de alcance que indique en qué experimento se usa cada familia. Así la tabla conecta el capítulo completo sin borrar las diferencias entre protocolos.

## Verificación realizada

- Benchmarks: `src/configs/benchmark_tuning_specs.py`, función `get_default_models`, define Dummy, GP lineal, GP RBF, GP Matérn \(3/2\) y GP Matérn \(5/2\). Si la dimensión es mayor que uno, añade ARD solo para RBF y Matérn \(5/2\). No define GP compuesto.
- Caso real: `src/configs/tuning_specs.py` define como modelos base Dummy, GP lineal, GP RBF, GP Matérn \(3/2\), GP Matérn \(5/2\) y GP compuesto. ARD se añade de forma selectiva para RBF, Matérn \(3/2\) o Matérn \(5/2\), según objetivo y representación de entrada.
- La salida `outputs/reports/TFG_MAIN_real_case_audit_no_tpc/active_model_metrics.csv` confirma que el caso real incluye `GP_Compuesto_NoARD` y no incluye una variante compuesta con ARD en el protocolo final.

## Ubicación

Archivo de referencia: `TFG_José/chapters/03_metodologia.tex`.

Secciones afectadas:

- `\section{Marco común de modelado sustituto}`
- `\subsection{Configuración experimental}` de benchmarks sintéticos
- `\subsection{Modelos, ajuste y métricas}` del caso real

## Decisión de estructura

El cuadro debe ir en el marco común, justo después de explicar que se compara una línea base constante con varias configuraciones de GP. Pero el cuadro no debe llamarse ni leerse como "modelos comunes a ambos experimentos", sino como "familias de modelos consideradas en el capítulo".

En el caso real no conviene repetir la tabla completa. Basta con remitir al cuadro común y explicar qué cambia en ese experimento: LODO anidado, dos representaciones de entrada, presencia del kernel compuesto y ARD selectiva.

## Cambio 1: sustituir la definición actual de Dummy y GP en el marco común

Sustituir desde:

```latex
El modelo Dummy actúa como referencia mínima.
```

hasta:

```latex
Esta descripción define el protocolo experimental; las clases y paquetes concretos empleados para ejecutarlo se describen por separado.
```

por:

```latex
El diseño compara una línea base constante y varias familias de GP. El Cuadro~\ref{tab:modelos_capitulo} resume qué modelos se usan en cada experimento y qué papel cumplen dentro del análisis. La línea base se interpreta como referencia mínima; los GP constituyen los modelos sustitutos probabilísticos analizados. Esta descripción define el protocolo experimental; las clases y paquetes concretos empleados para ejecutarlo se describen por separado.

\begin{table}[htbp]
    \centering
    \small
    \caption[Familias de modelos por experimento]{Familias de modelos consideradas por experimento.}
    \label{tab:modelos_capitulo}
    \renewcommand{\arraystretch}{1.12}
    \begin{tabular}{p{0.20\textwidth}p{0.22\textwidth}p{0.48\textwidth}}
        \toprule
        \textbf{Familia} & \textbf{Uso} & \textbf{Papel en el análisis} \\
        \midrule
        Dummy &
        Benchmarks y caso real &
        Predicción constante calculada a partir de las respuestas del conjunto de entrenamiento. Sirve como referencia mínima frente a modelos que sí utilizan las variables de entrada. \\

        GP lineal &
        Benchmarks y caso real &
        GP con kernel lineal y término de ruido blanco. Representa una hipótesis de relación aproximadamente lineal entre entradas y respuesta. \\

        GP RBF &
        Benchmarks y caso real &
        GP con kernel RBF y término de ruido blanco. Introduce una hipótesis suave y no lineal con una escala de longitud común. \\

        GP Matérn \(3/2\) &
        Benchmarks y caso real &
        GP con kernel Matérn \(3/2\) y término de ruido blanco. Permite funciones menos suaves que RBF. \\

        GP Matérn \(5/2\) &
        Benchmarks y caso real &
        GP con kernel Matérn \(5/2\) y término de ruido blanco. Supone una respuesta más suave que Matérn \(3/2\), pero menos restrictiva que RBF. \\

        GP compuesto &
        Caso real &
        GP con kernel aditivo lineal + Matérn \(5/2\) + ruido blanco. Se usa para combinar una tendencia global con una componente no lineal suave en el conjunto experimental de dietas. \\

        Variante ARD &
        Según protocolo &
        Variante con una escala de longitud por variable. En benchmarks se aplica a RBF y Matérn \(5/2\) cuando la dimensión lo permite; en el caso real se prueba solo sobre el mejor kernel previo de cada combinación de objetivo y representación de entrada. \\
        \bottomrule
    \end{tabular}
\end{table}
\FloatBarrier
```

## Cambio 2: ajustar la fila de modelos en benchmarks

Sustituir la fila:

```latex
        Modelos GP 
        & Kernels Matérn \(3/2\), Matérn \(5/2\), RBF y lineal; en dimensión mayor que uno se evalúan versiones isotrópicas y con ARD. \\
```

por:

```latex
        Modelos GP 
        & Familias lineal, RBF, Matérn \(3/2\) y Matérn \(5/2\), descritas en el Cuadro~\ref{tab:modelos_capitulo}; en dimensión mayor que uno se evalúan además variantes ARD para RBF y Matérn \(5/2\). \\
```

## Cambio 3: sustituir la tabla del caso real por una conexión al marco común

Sustituir desde:

```latex
En esta etapa se comparan una línea base simple y varias configuraciones de GP.
```

hasta el final del cuadro `tab:modelos_caso_real`.

por:

```latex
En esta etapa se reutilizan las familias del Cuadro~\ref{tab:modelos_capitulo} que corresponden al caso real: Dummy, GP lineal, GP RBF, GP Matérn \(3/2\), GP Matérn \(5/2\) y GP compuesto. Todos los modelos se evalúan bajo el protocolo LODO anidado descrito en la Sección~\ref{subsec:lodo_caso_real}, de forma independiente para cada objetivo y para cada representación de entrada. Para evitar una exploración excesiva de variantes, ARD no se aplica a todas las familias desde el inicio, sino solo al mejor kernel previo de cada combinación de objetivo y representación de entrada; en el protocolo final, esta variante se limita a RBF o Matérn, no al kernel compuesto.
```

## Relación con otras partes

- La definición de modelos queda en un único lugar, pero ahora distingue explícitamente qué se usa en benchmarks y qué se usa en el caso real.
- El caso real queda conectado con el marco común y con LODO, sin repetir una tabla completa.
- La nomenclatura evita `NoARD`. Cuando no hay ARD, se omite; cuando sí hay ARD, se menciona porque cambia la interpretación del modelo.
- Si se aplica este cambio, la etiqueta `tab:modelos_caso_real` desaparece y las referencias deben pasar a `tab:modelos_capitulo`.

## Notas fuera del LaTeX

No tocar la explicación de los pasos de limpieza y preprocesamiento por esta anotación. La duda afecta a la ubicación conceptual de las familias de modelos y a la nomenclatura ARD.
