# Anotación 04: quitar repetición sobre Dummy en métricas

## Anotación o motivo

La definición de Dummy ya queda en el marco común. En la sección de métricas de benchmarks no conviene repetir su función ni adelantar una interpretación que se puede explicar después con los resultados.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsection{Métricas del experimento benchmark}`.

## Rango de sustitución

Sustituir:

```latex
La capacidad predictiva se analiza mediante el MAE relativo respecto al estado inicial y mediante la comparación frente al modelo Dummy. El MAE relativo permite comprobar si las observaciones añadidas reducen el error del GP sobre un conjunto de test independiente. La comparación frente a Dummy se utiliza como referencia mínima para verificar que el GP aporta información más allá de una predicción trivial. Estas métricas deben separarse de la mejora relativa del mínimo encontrado: una configuración puede predecir mejor en promedio sobre el conjunto de test sin ser necesariamente la más útil para encontrar mínimos.
```

## Texto propuesto

```latex
La capacidad predictiva se analiza mediante el MAE relativo respecto al estado inicial y mediante la comparación frente al modelo Dummy. El MAE relativo permite comprobar si las observaciones añadidas reducen el error del GP sobre un conjunto de test independiente.
```

## Relación con otras partes

- La definición de Dummy debe quedar solo en `\subsection{Datos, modelos y preprocesamiento}`.
- La diferencia entre buen ajuste predictivo y utilidad para optimizar se puede comentar en resultados, donde ya aparecen los casos concretos.

## Notas

El cambio elimina repetición y deja la interpretación para la sección de resultados.
