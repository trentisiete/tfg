# Anotación 10: explicar el bloque de Hermetia sin usar el nombre del archivo

## Anotación o motivo

El nombre interno del archivo no aporta información al lector. Conviene explicar directamente con qué datos se trabaja: ensayos de \textit{Hermetia illucens}, dietas, réplicas y variables disponibles.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsection{Alcance del caso real y selección de \textit{Hermetia}}`.

## Rango de sustitución

Sustituir:

```latex
El archivo experimental original contiene información para dos especies: \textit{Hermetia illucens} y \textit{Tenebrio molitor}. Además, recoge varios bloques de información: composición nutricional y TPC de las dietas, parámetros productivos de la cría, y composición nutricional y TPC de las larvas obtenidas. Sin embargo, los resultados modelados en este experimento se limitan al dataset \texttt{productivity\_hermetia\_lote.csv}.
```

## Texto propuesto

```latex
El archivo experimental original contiene información para dos especies: \textit{Hermetia illucens} y \textit{Tenebrio molitor}. Además, recoge varios bloques de información: composición nutricional y TPC de las dietas, parámetros productivos de la cría, y composición nutricional y TPC de las larvas obtenidas. En este experimento se trabaja únicamente con el bloque experimental de \textit{Hermetia illucens}.
```

## Relación con otras partes

- Esta propuesta debe aplicarse junto con la anotación 11, que concreta por qué no se usa \textit{Tenebrio molitor}.
- El nombre del archivo también debe eliminarse del Cuadro del dataset; se propone en la anotación 15.

## Notas fuera del LaTeX

Se mantiene la explicación de las dos especies, pero se sustituye la referencia técnica al CSV por una descripción comprensible para quien no ve los datos.
