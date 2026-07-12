# Anotación 18: definir el kernel compuesto antes de los resultados

## Anotación o motivo

La fórmula del kernel compuesto aparece actualmente dentro de la subsección de resultados `\subsubsection{Expresividad geométrica del GP compuesto}`, después de la figura que compara familias de kernel. La observación de la profesora es acertada: si el texto habla del kernel compuesto como modelo evaluado, su estructura debe quedar definida antes de interpretar sus resultados.

## Ubicación

Archivo de referencia: `TFG_José/chapters/03_metodologia.tex`.

Ubicación recomendada: `\subsection{Modelos, ajuste y métricas}`, justo después del párrafo que enumera los modelos usados en el caso real y antes de `El ajuste se organiza en tres niveles:`.

Si se aplica la anotación 17, este texto debe ir después del párrafo que remite al `Cuadro~\ref{tab:modelos_capitulo}`. El kernel compuesto no debe definirse en el marco común como si se usara también en benchmarks: es específico del caso real.

## Cambio 1: añadir la definición en modelos del caso real

Insertar antes de:

```latex
El ajuste se organiza en tres niveles:
```

el siguiente texto:

```latex
En el caso real, el GP compuesto se concreta mediante una suma de kernels:
\[
k(\bm{x},\bm{x}')
=
C_1\,k_{\text{Lineal}}(\bm{x},\bm{x}')
+
C_2\,k_{\text{Matérn }5/2}(\bm{x},\bm{x}')
+
k_{\text{White}}(\bm{x},\bm{x}'),
\]
donde \(C_1\) y \(C_2\) son constantes de escala positivas. La componente lineal recoge una tendencia global de la respuesta respecto a las variables de entrada; la componente Matérn \(5/2\) permite desviaciones locales suaves; y el término de ruido blanco representa variabilidad experimental no explicada por las covariables disponibles.
```

## Cambio 2: eliminar la primera definición tardía en resultados

En `\subsubsection{Expresividad geométrica del GP compuesto}`, eliminar desde:

```latex
El kernel compuesto utilizado puede escribirse de forma esquemática como
```

hasta:

```latex
El kernel compuesto actúa como un compromiso: mantiene una componente interpretable de tendencia global y añade flexibilidad local para modelar desviaciones suaves alrededor de esa tendencia.
```

Sustituirlo por:

```latex
Esta lectura es coherente con la definición previa del GP compuesto: una estructura de covarianza que combina una tendencia global, flexibilidad local y ruido experimental. Por ello, la comparación de la Figura~\ref{fig:kernel_family_mae} no enfrenta solo nombres de modelos, sino hipótesis distintas sobre la forma de la relación entre dieta y respuesta.
```

## Relación con otras partes

- La explicación formal del modelo queda en la parte metodológica, antes de los resultados.
- La subsección de resultados queda centrada en interpretar la comparación empírica, no en definir por primera vez el modelo.
- La propuesta es compatible con la anotación 17: el cuadro del marco común indica que el GP compuesto pertenece al caso real, y aquí se concreta su fórmula.

## Notas fuera del LaTeX

No mover la fórmula al bloque común de benchmarks. El compuesto no se usa en benchmarks; solo debe aparecer como familia específica del caso real.
