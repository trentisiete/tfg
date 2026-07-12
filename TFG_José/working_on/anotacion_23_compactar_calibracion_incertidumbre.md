# Anotación 23: compactar la calibración de incertidumbre

## Anotación o motivo

La subsección de calibración repite varias veces la misma conclusión: la incertidumbre del GP aporta información útil, pero no está perfectamente calibrada y no debe verse como garantía. Esa idea es importante, pero puede decirse una vez de forma clara.

## Prioridad

Media. Ahorra varias líneas y mejora el ritmo de lectura.

## Ubicación

Archivo: `TFG_José/chapters/03_metodologia.tex`.

Sección: `\subsubsection{Calibración práctica de la incertidumbre}`.

## Rango de sustitución

Sustituir desde:

```latex
Además de evaluar la precisión de la media predictiva, en este caso real interesa comprobar si la incertidumbre del GP aporta información útil sobre la fiabilidad de sus predicciones.
```

hasta antes de:

```latex
\begin{figure}[htbp]
```

por:

```latex
Además de evaluar la precisión de la media predictiva, interesa comprobar si la incertidumbre del GP ayuda a identificar predicciones menos fiables. La Figura~\ref{fig:uncertainty_vs_error} compara, para cada objetivo, la desviación estándar predictiva con el error absoluto observado en las dietas retenidas por LODO.

La relación es más clara en quitina, donde el modelo tiende a asignar mayor incertidumbre a observaciones con errores más altos. En FCR y proteína, en cambio, la calibración es más débil: el GP no ordena de forma consistente los casos fáciles y difíciles según su desviación estándar. Por tanto, la incertidumbre se interpreta como una señal de apoyo, útil pero no uniforme, y no como una garantía de fiabilidad.
```

Después de la figura, sustituir:

```latex
En consecuencia, la incertidumbre del GP se interpreta en este trabajo como una señal de apoyo para la toma de decisiones, no como una garantía absoluta de fiabilidad. Su papel es especialmente útil para priorizar ensayos futuros y detectar regiones donde el modelo reconoce mayor desconocimiento. Sin embargo, cualquier candidato propuesto a partir del sustituto debe entenderse como una hipótesis experimental pendiente de validación, no como una conclusión cerrada sobre el rendimiento real de una nueva dieta.
```

por:

```latex
Una visualización complementaria de los intervalos predictivos por dieta se recoge en el Anexo~\ref{app:intervalos_prediccion_dieta}.
```

## Ahorro estimado

5--7 líneas.

## Relación con otras partes

- La idea de "hipótesis experimental pendiente de validación" se conserva mejor en la subsección de EI, donde realmente se habla de candidatos.
- La calibración queda centrada en la lectura de la figura y no anticipa en exceso la discusión prospectiva.

## Notas fuera del LaTeX

No se pierde el mensaje de cautela; queda condensado en el segundo párrafo.
