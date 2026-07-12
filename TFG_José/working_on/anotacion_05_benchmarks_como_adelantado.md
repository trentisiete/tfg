# Anotación 05: introducir benchmarks sin repetir

## Anotación o motivo

La idea del experimento con benchmarks ya se ha explicado al inicio del capítulo. Conviene marcarlo con una transición y acortar la explicación.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\section{Experimento 1: benchmarks sintéticos}`.

## Rango de sustitución

Sustituir:

```latex
El primer experimento se desarrolla sobre funciones benchmark sintéticas, donde la función objetivo puede evaluarse en nuevos puntos del dominio. Esto permite analizar el ciclo completo de optimización basada en modelos sustitutos: partir de un diseño inicial, ajustar un GP, seleccionar nuevas evaluaciones mediante EI y actualizar secuencialmente el conjunto observado. El objetivo no es solo medir el error final del modelo, sino estudiar cómo evolucionan el mejor valor encontrado, la capacidad predictiva y la incertidumbre a medida que se incorporan nuevas observaciones.
```

## Texto propuesto

```latex
Como se ha adelantado, el primer experimento utiliza funciones benchmark sintéticas, en las que la función objetivo puede evaluarse en nuevos puntos del dominio. Esto permite estudiar el ciclo completo de optimización basada en modelos sustitutos: diseño inicial, ajuste del GP, selección de nuevas evaluaciones mediante EI y actualización secuencial del conjunto observado. La lectura se centra en la evolución del mejor valor encontrado, la capacidad predictiva y la incertidumbre conforme se incorporan nuevas observaciones.
```

## Relación con otras partes

- El inicio del capítulo ya distingue benchmarks y caso real.
- La sección de métricas concreta después cómo se mide cada parte.

## Notas

Esta propuesta sustituye la explicación repetida por una entrada más breve.
