# Anotación 15: quitar el nombre del CSV en el cuadro del caso real

## Anotación o motivo

El nombre `productivity_hermetia_lote.csv` es interno y no aporta información al lector. El cuadro debe describir el bloque de datos usado, no el archivo.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Cuadro `tab:dataset_caso_real`.

## Rango de sustitución

Sustituir el cuadro completo:

```latex
\begin{table}[htbp]
    \centering
    \small
    \caption[Dataset del caso real]{Resumen del dataset utilizado en el caso real.}
    \label{tab:dataset_caso_real}
    \renewcommand{\arraystretch}{1.12}
    \begin{tabular}{ll}
        \toprule
        \textbf{Elemento} & \textbf{Descripción} \\
        \midrule
        Dataset modelado & \texttt{productivity\_hermetia\_lote.csv} \\
        Especie & \textit{Hermetia illucens} \\
        Observaciones & 33 réplicas experimentales \\
        Dietas & 11 dietas \\
        Réplicas por dieta & 3 \\
        Dieta de referencia & \texttt{Control}, basada en salvado de trigo \\
        Subproductos principales & Hoja de olivo, orujo de oliva y cascarilla de quinoa \\
        Variable de grupo & \texttt{diet\_name} \\
        Objetivos & \texttt{FCR}, \texttt{QUITINA (\%)}, \texttt{PROTEINA (\%)} \\
        Nivel de trabajo & Réplica experimental \\
        \bottomrule
    \end{tabular}
\end{table}
```

## Texto propuesto

```latex
\begin{table}[htbp]
    \centering
    \small
    \caption[Datos del caso real]{Resumen de los datos utilizados en el caso real.}
    \label{tab:dataset_caso_real}
    \renewcommand{\arraystretch}{1.12}
    \begin{tabular}{ll}
        \toprule
        \textbf{Elemento} & \textbf{Descripción} \\
        \midrule
        Bloque analizado & Ensayos de \textit{Hermetia illucens} \\
        Observaciones & 33 réplicas experimentales \\
        Dietas evaluadas & 11 formulaciones \\
        Réplicas por dieta & 3 lotes independientes \\
        Dieta de referencia & Salvado de trigo sin inclusión de subproductos \\
        Subproductos principales & Hoja de olivo, orujo de oliva y cascarilla de quinoa \\
        Agrupación para validación & Réplicas de una misma dieta \\
        Objetivos modelados & FCR, quitina y proteína larvaria \\
        Nivel de trabajo & Réplica experimental \\
        \bottomrule
    \end{tabular}
\end{table}
```

## Relación con otras partes

- El texto anterior al cuadro debe referirse a él como `Cuadro~\ref{tab:dataset_caso_real}`.
- La tabla queda más comprensible para quien no conoce el archivo procesado.

## Notas fuera del LaTeX

Se eliminan también `diet_name` y los nombres exactos de columnas de objetivos.
