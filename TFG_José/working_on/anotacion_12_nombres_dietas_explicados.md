# Anotación 12: usar nombres de dietas solo si se explican

## Anotación o motivo

Los nombres como `Hoja15` se usan después en resultados, tablas y figuras. Por tanto, conviene mantenerlos, pero explicarlos como etiquetas abreviadas y no como si el lector conociera la tabla original.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsection{Diseño experimental y dietas consideradas}`.

## Rango de sustitución

Sustituir:

```latex
El dataset modelado contiene una dieta control y varias dietas no convencionales. La dieta \texttt{Control} corresponde al salvado de trigo sin inclusión de subproductos y actúa como referencia experimental. Las dietas no convencionales se nombran combinando el subproducto utilizado y su porcentaje de inclusión. Así, por ejemplo, \texttt{Hoja15}, \texttt{Hoja30} y \texttt{Hoja50} indican dietas con 15 \%, 30 \% y 50 \% de hoja de olivo, respectivamente. La misma lógica se aplica a las dietas con orujo de oliva y cascarilla de quinoa.
```

## Texto propuesto

```latex
El conjunto analizado contiene una dieta control, basada en salvado de trigo, y varias dietas no convencionales en las que parte de ese salvado se sustituye por hoja de olivo, orujo de oliva o cascarilla de quinoa. Para identificar las formulaciones se usan etiquetas breves que combinan el subproducto y el porcentaje de inclusión: por ejemplo, \texttt{Hoja15} indica una dieta con un 15\,\% de hoja de olivo. La misma regla se aplica a las dietas con orujo y quinoa.
```

## Relación con otras partes

- Mantener estas etiquetas es útil porque aparecen después en el mapa de errores y en los candidatos prospectivos.
- La explicación debe aparecer una sola vez; después basta con usar las etiquetas.

## Notas fuera del LaTeX

La propuesta conserva nombres como `Hoja15`, pero reduce la lista de ejemplos para ahorrar espacio.
