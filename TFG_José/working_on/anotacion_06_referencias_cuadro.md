# Anotación 06: usar "Cuadro" en referencias a tablas

## Anotación o motivo

La plantilla muestra las tablas como "Cuadro" en el documento final. Por coherencia, las referencias textuales deben usar "Cuadro" en lugar de "Tabla".

## Diagnóstico

En `TFG_José/working_on/cap3.tex` aparecen estas referencias con `Tabla~\ref{...}`:

- `tab:app_dominio_borehole`
- `tab:app_semillas_benchmarks`
- `tab:resumen_incumbent_benchmarks`
- `tab:resumen_predictivo_benchmarks`
- `tab:jerarquia_factores_benchmark`
- `tab:incumbents_candidatos_ei`

Además, en las propuestas anteriores hay que usar también `Cuadro~\ref{...}` para las tablas nuevas o ya detectadas.

## Cambio 1

Sustituir:

```latex
El dominio detallado de Borehole, el flujo completo del proceso de \emph{infill} y la configuración específica de semillas se incluyen como material complementario en el Anexo~\ref{app:benchmarks_complementario}. En particular, el dominio de Borehole se recoge en la Tabla~\ref{tab:app_dominio_borehole}, el esquema del proceso secuencial en la Figura~\ref{fig:app_flujo_infill} y las semillas empleadas en la Tabla~\ref{tab:app_semillas_benchmarks}.
```

por:

```latex
El dominio detallado de Borehole, el flujo completo del proceso de \emph{infill} y la configuración específica de semillas se incluyen como material complementario en el Anexo~\ref{app:benchmarks_complementario}. En particular, el dominio de Borehole se recoge en el Cuadro~\ref{tab:app_dominio_borehole}, el esquema del proceso secuencial en la Figura~\ref{fig:app_flujo_infill} y las semillas empleadas en el Cuadro~\ref{tab:app_semillas_benchmarks}.
```

## Cambio 2

Sustituir:

```latex
La Tabla~\ref{tab:resumen_incumbent_benchmarks} resume el mejor modelo de cada benchmark según la mejora final del mínimo encontrado. También se incluye la mejora acumulada, que permite distinguir si la mejora aparece pronto durante la trayectoria o si se concentra al final del presupuesto de \emph{infill}.
```

por:

```latex
El Cuadro~\ref{tab:resumen_incumbent_benchmarks} resume el mejor modelo de cada benchmark según la mejora final del mínimo encontrado. También se incluye la mejora acumulada, que permite distinguir si la mejora aparece pronto durante la trayectoria o si se concentra al final del presupuesto de \emph{infill}.
```

## Cambio 3

Sustituir:

```latex
La Tabla~\ref{tab:resumen_predictivo_benchmarks} resume el mejor modelo predictivo de cada benchmark y su comparación frente a Dummy. Dummy no participa en el proceso de \emph{infill}, pero sirve como referencia para comprobar si el GP aporta valor frente a una predicción trivial.
```

por:

```latex
El Cuadro~\ref{tab:resumen_predictivo_benchmarks} resume el mejor modelo predictivo de cada benchmark y su comparación frente a Dummy. Dummy no participa en el proceso de \emph{infill}, pero sirve como referencia para comprobar si el GP aporta valor frente a una predicción trivial.
```

## Cambio 4

Sustituir:

```latex
La Tabla~\ref{tab:jerarquia_factores_benchmark} no debe interpretarse como un ranking estadístico estricto, sino como una síntesis empírica de los patrones
```

por:

```latex
El Cuadro~\ref{tab:jerarquia_factores_benchmark} no debe interpretarse como un ranking estadístico estricto, sino como una síntesis empírica de los patrones
```

## Cambio 5

Sustituir:

```latex
La Tabla~\ref{tab:incumbents_candidatos_ei} compara, para cada objetivo, el mejor punto observado experimentalmente con el candidato no ensayado priorizado por EI. Esta distinción es importante: el \emph{incumbent} observado es una evidencia experimental, mientras que el candidato EI es una hipótesis propuesta por el modelo para una posible evaluación posterior.
```

por:

```latex
El Cuadro~\ref{tab:incumbents_candidatos_ei} compara, para cada objetivo, el mejor punto observado experimentalmente con el candidato no ensayado priorizado por EI. Esta distinción es importante: el \emph{incumbent} observado es una evidencia experimental, mientras que el candidato EI es una hipótesis propuesta por el modelo para una posible evaluación posterior.
```

## Relación con otras partes

- La anotación 03 también debe usar `Cuadro~\ref{...}` en las referencias nuevas que propone.
- Las referencias a figuras se mantienen como `Figura~\ref{...}`.

## Notas

Es un ajuste terminológico, no conceptual.
