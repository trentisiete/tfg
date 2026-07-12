# Anotación 22: compactar la generalización LODO sin perder el hilo

## Anotación o motivo

La idea "LODO evalúa dietas completas no vistas" aparece en el diseño de dietas, en el preprocesamiento, en el protocolo LODO, en modelos y métricas, y vuelve a explicarse de forma extensa al inicio de resultados. En resultados conviene recordarla, pero no repetir todo el mecanismo.

## Prioridad

Media-alta. Ahorra espacio y hace la lectura menos pesada sin reducir comprensión, porque el protocolo LODO ya está definido antes.

## Ubicación

Archivo: `TFG_José/chapters/03_metodologia.tex`.

Sección: `\subsubsection{Generalización estructural frente a memorización}`.

## Cambio 1: compactar la entrada de la subsección

Sustituir:

```latex
La primera cuestión es si el modelo sustituto es capaz de extraer una señal general a partir de las variables de formulación y composición, o si su rendimiento depende únicamente de reconocer dietas ya presentes en el entrenamiento. Para responder a esta pregunta se utiliza el protocolo LODO descrito en la Sección~\ref{subsec:lodo_caso_real}. En cada fold externo se deja fuera una dieta completa, de modo que ninguna de sus réplicas participa ni en el ajuste del modelo ni en la selección de hiperparámetros. Así, la evaluación reproduce una situación más exigente que una partición aleatoria por filas: el modelo debe predecir la respuesta de una formulación no vista.
```

por:

```latex
La primera cuestión es si el modelo sustituto extrae una señal general de las variables de formulación y composición, o si solo reconoce dietas similares a las observadas. Para ello se utiliza el protocolo LODO descrito en la Sección~\ref{subsec:lodo_caso_real}, que obliga al modelo a predecir una formulación completa no vista.
```

## Cambio 2: compactar la interpretación de las dos primeras figuras

Sustituir desde:

```latex
La Figura~\ref{fig:lodo_generalization_vs_dummy} resume el rendimiento relativo frente al modelo Dummy.
```

hasta:

```latex
entonces parte de la información aprendida procede de relaciones compartidas entre dietas, y no solo de la repetición de observaciones muy similares dentro de una misma formulación.
```

por:

```latex
La Figura~\ref{fig:lodo_generalization_vs_dummy} resume el rendimiento relativo frente a Dummy: valores inferiores a \(1\) indican menor MAE que la predicción trivial. Bajo esta lectura, el GP compuesto mejora al Dummy en los tres objetivos activos, con reducciones relativas del \(15.8\%\) en FCR, \(15.4\%\) en quitina y \(36.6\%\) en proteína. Por tanto, el modelo aprovecha información de las variables de entrada y no se limita a reproducir una media global.

La Figura~\ref{fig:error_by_left_out_diet} matiza esta mejora agregada: la dificultad cambia según la dieta retenida. En FCR destacan \texttt{Hoja50} y algunas formulaciones con orujo o quinoa; en quitina, \texttt{Orujo70}, \texttt{Orujo90} y \texttt{Control}; y en proteína, \texttt{Orujo50} y algunas formulaciones con hoja. En conjunto, el GP compuesto captura regularidades útiles, pero la señal aprendida no se transfiere con la misma calidad a todas las dietas.
```

## Ahorro estimado

5--8 líneas, manteniendo las cifras principales y la lectura de ambas figuras.

## Relación con otras partes

- El detalle de LODO se conserva en su sección metodológica.
- Se mantiene la conclusión esencial: mejora frente a Dummy, pero generalización no uniforme.
- Se elimina la repetición de "réplicas", "partición aleatoria por filas" y "dieta completa no vista" dentro de resultados.

## Notas fuera del LaTeX

Si se aplica también la anotación 20, el segundo párrafo puede ajustarse para mencionar que el color de la figura expresa dificultad relativa por objetivo.
