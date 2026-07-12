# Anotación 08: matizar la fila de kernel en el cuadro de factores

## Anotación o motivo

La anotación cuestiona la frase que enumera un único mejor kernel por benchmark. En los datos agregados hay casos muy próximos, especialmente en Borehole, por lo que no conviene presentar esos resultados como ganadores claros. La conclusión más sólida es que RBF y Matérn funcionan bien de forma general, y que la elección depende del benchmark.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsubsection{Influencia de las condiciones experimentales}`, fila `Kernel` del cuadro `tab:jerarquia_factores_benchmark`.

## Rango de sustitución

Sustituir la fila:

```latex
  3
  & Kernel
  & El mejor kernel cambia por benchmark: \(GP\_Linear\) en Borehole, \(GP\_Matern52\_ARD\) en Branin, \(GP\_Matern52\) en Forrester y \(GP\_RBF\) en Hartmann6.
  & El kernel importa, pero no hay una familia universalmente ganadora. La elección útil depende de la geometría de la función y del objetivo: predecir bien no siempre
  equivale a encontrar mejor \emph{incumbent}. \\
```

## Texto propuesto

```latex
  3
  & Kernel
  & Las diferencias entre kernels no siempre son concluyentes. En Borehole varios modelos quedan prácticamente empatados, mientras que en Branin, Forrester y Hartmann6 destacan sobre todo variantes RBF y Matérn.
  & El kernel importa, pero no debe leerse como una competición con un único ganador. RBF y Matérn ofrecen el comportamiento más consistente, aunque la elección final depende de la geometría del benchmark y del criterio analizado. \\
```

## Relación con otras partes

- Encaja con la anotación 07: RBF y Matérn se pueden presentar como familias más estables sin afirmar que ganen siempre.
- Evita contradicción con el anexo, donde algunas diferencias visuales son pequeñas.

## Notas fuera del LaTeX

En la tabla de mejora del \emph{incumbent}, Borehole tiene valores casi idénticos para lineal, Matérn y RBF. Por eso es mejor hablar de empate práctico o diferencias no concluyentes.
