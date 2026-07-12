# Anotación 02: reducir repetición en la introducción de benchmarks

## Anotación o motivo

La profesora señala que la idea del primer experimento ya se ha adelantado varias veces. Conviene reconocerlo con una transición breve y condensar el párrafo para no repetir la explicación completa.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\section{Experimento 1: benchmarks sintéticos}`.

## Rango de sustitución

Sustituir este párrafo completo:

```latex
El primer experimento se desarrolla sobre funciones benchmark sintéticas, donde la función objetivo puede evaluarse en nuevos puntos del dominio. Esto permite analizar el ciclo completo de optimización basada en modelos sustitutos: partir de un diseño inicial, ajustar un GP, seleccionar nuevas evaluaciones mediante EI y actualizar secuencialmente el conjunto observado. El objetivo no es solo medir el error final del modelo, sino estudiar cómo evolucionan el mejor valor encontrado, la capacidad predictiva y la incertidumbre a medida que se incorporan nuevas observaciones.
```

## Texto propuesto

```latex
Como se ha adelantado, el primer experimento utiliza funciones benchmark sintéticas, en las que la función objetivo puede evaluarse en nuevos puntos del dominio. Esto permite estudiar el ciclo completo de optimización basada en modelos sustitutos: diseño inicial, ajuste del GP, selección de nuevas evaluaciones mediante EI y actualización secuencial del conjunto observado. La lectura se centra en la evolución del mejor valor encontrado, la capacidad predictiva y la incertidumbre conforme se incorporan nuevas observaciones.
```

## Relación con otras partes

- El capítulo ya introduce los dos escenarios experimentales en los primeros párrafos, especialmente al explicar la diferencia entre benchmarks sintéticos y caso real.
- La Sección `\ref{subsubsec:metricas_benchmarks}` vuelve a separar error predictivo, mejora del mínimo encontrado e incertidumbre. Por eso esta introducción debe funcionar como puente, no como explicación exhaustiva.

## Notas

El cambio mantiene el contenido, pero añade la transición pedida y reduce la sensación de repetición.
