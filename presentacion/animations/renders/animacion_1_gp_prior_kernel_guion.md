# Guion - Animacion 1: prior GP y kernel RBF

Esta animacion introduce la idea de un Proceso Gaussiano como una distribucion sobre funciones, no como una unica funcion concreta.

Al inicio aparece:

```latex
f \sim \mathcal{GP}(0, k_\ell)
```

Idea a explicar:

> Un Proceso Gaussiano define una distribucion sobre posibles funciones. En este caso asumimos media cero y un kernel \(k_\ell\), que determina que formas de funcion consideramos plausibles antes de observar datos.

Cuando aparecen varias curvas con \(\ell = 0.30\):

> Cada curva es una muestra distinta del prior. Todavia no hemos visto datos, asi que no estamos ajustando nada. Solo estamos viendo que tipo de funciones permite nuestra hipotesis previa.

Cuando aparece el kernel RBF:

```latex
k_\ell(x,x') = \sigma^2 \exp\left(-\frac{(x-x')^2}{2\ell^2}\right)
```

Idea a explicar:

> El kernel mide cuanto se relacionan dos puntos de entrada \(x\) y \(x'\). Si estan cerca, sus valores de funcion tienden a estar mas correlacionados. Si estan lejos, la correlacion cae.

Cuando aparece \(x_0\) y la campana inferior:

> Esta curva amarilla muestra la correlacion entre un punto fijo \(x_0\) y el resto del dominio. No es la funcion objetivo; es la forma en la que el kernel propaga dependencia alrededor de un punto.

Al pasar de \(\ell = 0.30\) a \(\ell = 1.15\):

> Al aumentar \(\ell\), la correlacion se extiende a una region mas amplia. Como puntos mas alejados siguen estando relacionados, las funciones muestreadas cambian de forma mas gradual.

En el cierre:

```latex
\ell \uparrow \Rightarrow \text{correlacion} \uparrow \Rightarrow \text{suavidad} \uparrow
```

Resumen oral:

> La longitud de escala controla la suavidad del prior. No es un simple parametro numerico: codifica una hipotesis sobre como esperamos que se comporte la funcion antes de medirla.

## Insights para la defensa

Frase principal:

> El kernel es donde introducimos conocimiento previo en el modelo. En un GP, no solo elegimos una tecnica de regresion; elegimos una hipotesis sobre la geometria de las funciones posibles.

Relacion con modelos sustitutos:

> Esto es especialmente importante en modelos sustitutos, porque normalmente trabajamos con pocos datos. Si las evaluaciones son caras, el prior y el kernel tienen mucho peso en las predicciones iniciales.

Matiz sobre el hiperparametro:

> Una \(\ell\) demasiado pequena permite funciones muy flexibles, pero puede sobreinterpretar variaciones locales. Una \(\ell\) demasiado grande impone suavidad excesiva y puede ocultar cambios importantes. Por eso ajustar o validar este hiperparametro es clave.
