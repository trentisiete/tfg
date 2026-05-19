# Estado del arte y fundamentos

## Optimización global con evaluaciones costosas y/o ruidosas {#sec:global_expensive_noisy}

Muchos problemas reales de optimización en ingeniería y ciencia de datos
se caracterizan por tener una función objetivo de *caja negra* y
evaluaciones costosas. En estos escenarios, evaluar una configuración de
entradas $\bm{x}$ no significa calcular una fórmula cerrada, sino
ejecutar un proceso externo (p. ej., una simulación numérica o un
experimento físico) que devuelve una salida de interés. En casos
industriales se han reportado tiempos de evaluación de decenas de horas
para una única simulación de alta complejidad, lo que hace inviable
realizar miles de evaluaciones durante una optimización iterativa
[@jiang2020surrogate].

Formalmente, consideramos un espacio de búsqueda continuo
$\mathcal{X}\subset\mathbb{R}^{d}$ y una función objetivo desconocida
$f:\mathcal{X}\to\mathbb{R}$. El objetivo de la *optimización global* es
encontrar un minimizador global
$$\bm{x}^{\star} \in \arg\min_{\bm{x}\in\mathcal{X}} f(\bm{x}).
\label{eq:global_opt_problem}$$ La dificultad central es que $f(\bm{x})$
no se evalúa de manera barata. En su lugar, cada consulta suele implicar
un coste $c(\bm{x})$ elevado, lo que impone un presupuesto máximo de
evaluaciones $B$. Por tanto, el conjunto de datos observable queda
restringido a un número pequeño de puntos,
$$\mathcal{D}_n = \{(\bm{x}_i, y_i)\}_{i=1}^{n}, \qquad n \le B,
\label{eq:dataset_budget}$$ donde $y_i$ denota el resultado observado al
evaluar en $\bm{x}_i$.

Además del coste, en muchos problemas las evaluaciones están afectadas
por *ruido*. En experimentos físicos, el ruido proviene de errores de
medida, variabilidad del entorno o diferencias entre réplicas. En
simulaciones complejas, puede aparecer *ruido numérico* debido a
múltiples causas como fenómenos estocásticos internos. Una forma
estándar de modelar esta situación es asumir que la observación es una
perturbación de la función subyacente:
$$y_i = f(\bm{x}_i) + \varepsilon_i, 
\qquad 
\mathbb{E}[\varepsilon_i]=0,
\qquad 
\mathrm{Var}(\varepsilon_i)=\sigma_\varepsilon^2.
\label{eq:noisy_observations}$$

Estas dos restricciones (presupuesto limitado y/o ruido) hacen que
muchos optimizadores clásicos resulten poco adecuados: los métodos
basados en gradientes requieren derivadas (normalmente no disponibles en
caja negra) y pueden ser inestables bajo ruido. En consecuencia, el
problema se transforma en una búsqueda bajo escasez de datos: con pocas
evaluaciones debemos (i) aproximar $f$ lo suficiente como para guiar la
búsqueda y (ii) decidir adaptativamente dónde invertir las siguientes
evaluaciones para mejorar la solución global.

Esta motivación conduce de forma natural al enfoque de *optimización
basada en modelos sustitutos* (SBO): construir un modelo estadístico
$\hat{f}$ a partir de $\mathcal{D}_n$ y emplearlo para proponer nuevas
evaluaciones, reduciendo de manera drástica el número de consultas
directas a la función costosa
[@forrester2008engineering; @jiang2020surrogate]. En las siguientes
secciones se introducen estos modelos y cómo se integran.

## Modelos sustitutos: definición y taxonomía {#sec:surrogates_def_tax}

En problemas de ingeniería y ciencia de datos, es frecuente que la
función objetivo sea una *caja negra*: la relación $y=f(\bm{x})$ no está
disponible en forma cerrada y, además, cada evaluación puede ser costosa
y/o ruidosa. En este contexto, un *modelo sustituto* (o *metamodelo*) es
una aproximación $\hat{f}$ aprendida a partir de un conjunto finito de
observaciones $\mathcal{D}_n=\{(\bm{x}_i,y_i)\}_{i=1}^n$, con el
objetivo de emular la respuesta del sistema real a un coste
computacional muy inferior. [@forrester2008engineering; @jiang2020]

### Propósito y rol dentro del proceso de diseño

Un modelo sustituto puede emplearse con dos fines principales:

1.  **Predicción rápida y análisis del espacio de diseño.** Una vez
    entrenado, $\hat{f}$ permite evaluar muchas configuraciones de
    entrada para comprender tendencias, analizar sensibilidad o explorar
    regiones de interés sin ejecutar el sistema costoso.

2.  **Optimización con presupuesto limitado.** En *surrogate-based
    optimization*, el sustituto se utiliza para guiar la selección de
    nuevas evaluaciones del sistema real, reduciendo el número de
    ejecuciones costosas necesarias para aproximarse a la solución
    óptima. [@jiang2020]

En este TFG, el foco principal es el segundo punto (optimización bajo
coste), sin perder la capacidad del sustituto como herramienta de
exploración e interpretación del problema.

### Taxonomía práctica de modelos sustitutos

Existe gran variedad de modelos que pueden actuar como sustitutos. Una
taxonomía orientada a la práctica, es la siguiente:

-   **Según la naturaleza de la predicción:**

    -   *Deterministas*, que producen una predicción puntual
        $\hat{f}(\bm{x})$.

    -   *Probabilísticos*, que además cuantifican incertidumbre (p. ej.
        mediante una distribución predictiva). Esta distinción es
        relevante cuando se quieren criterios de muestreo adaptativo
        basados en exploración/explotación.
        [@forrester2008engineering; @jiang2020]

-   **Según la flexibilidad del modelo:**

    -   *Paramétricos* (p. ej. regresiones polinómicas), simples e
        interpretables, pero potencialmente rígidos si la respuesta
        presenta multimodalidad o alta no linealidad.

    -   *No paramétricos* o de alta flexibilidad (p. ej. métodos kernel,
        árboles, ensambles), capaces de capturar respuestas complejas, a
        costa de más hiperparámetros y riesgo de sobreajuste con pocos
        datos. [@forrester2008engineering]

-   **Según cómo se integra con la optimización:**

    -   *Offline*: se entrena una única vez con $\mathcal{D}_n$ y se
        utiliza para proponer candidatos (útil cuando obtener nuevas
        evaluaciones es inviable en el corto plazo).

    -   *Secuencial/online*: el sustituto se actualiza iterativamente
        añadiendo nuevas evaluaciones seleccionadas de forma informada
        (infill/adaptive sampling), hasta agotar presupuesto o alcanzar
        un criterio de parada. [@forrester2008engineering; @jiang2020]

### Flujo de trabajo genérico de construcción y refinamiento

A pesar de la diversidad de modelos, el proceso general de modelado
sustituto sigue pasos estándar:

1.  **Definir variables y dominio.** Se establecen las variables de
    decisión (entradas) y sus rangos, fijando el dominio $\mathcal{X}$.

2.  **Muestreo inicial (DoE).** Se selecciona un conjunto inicial de
    puntos $\{\bm{x}_i\}_{i=1}^n$ que cubra el espacio de forma
    razonablemente uniforme. En problemas continuos es habitual usar
    muestreos *space-filling*, como Latin Hypercube Sampling (LHS) o
    SOBOL. [@forrester2008]

3.  **Evaluación del sistema real.** Se calculan las salidas $y_i$ con
    el sistema real y se construye el dataset inicial $\mathcal{D}_n$.

4.  **Ajuste y validación del sustituto.** Se elige una familia de
    modelos, se estiman sus parámetros y se evalúa su calidad predictiva
    con prácticas estándar (p. ej. validación cruzada), teniendo
    presente compromisos entre complejidad y robustez.
    [@forrester2008engineering]

5.  **Refinamiento secuencial: muestreo adaptativo / *active
    learning*.** Cuando el presupuesto lo permite, se añaden nuevos
    puntos (*infill points*) seleccionados por un criterio que
    prioriza: (i) regiones donde el modelo es probablemente inexacto
    (exploración), y/o (ii) regiones prometedoras respecto al óptimo
    (explotación). Este ciclo se repite hasta alcanzar un criterio de
    parada (presupuesto máximo, mejora marginal, o precisión objetivo).
    [@forrester2008; @jiang2020]

### Aplicación al marco experimental de este TFG

Este trabajo considerará dos escenarios complementarios. En los
*benchmarks*, la función objetivo está disponible para ser evaluada
tantas veces como se requiera, lo que permite estudiar de forma
controlada el ciclo secuencial completo (pasando de $n=0$ o $n$ con
muestreo inicial, a $n=N$ muestras) y analizar cómo evolucionan la
precisión del sustituto y las estrategias de muestreo adaptativo. En el
*caso real*, donde la obtención de nuevas observaciones requiere como
mínimo de 3 meses, el sustituto se utilizará en régimen
*offline/prospectivo*: entrenamiento con el dataset disponible,
evaluación interna del rendimiento y propuesta de configuraciones
candidatas $\bm{x}^\star$ para su posible validación futura.

## El Proceso Gaussiano como modelo sustituto {#sec:gp_surrogate}

Entre los modelos sustitutos probabilísticos, el Proceso Gaussiano (GP)
destaca por ofrecer un marco bayesiano elegante y flexible para la
regresión no paramétrica. A diferencia de los métodos que buscan unos
parámetros óptimos fijos, el GP infiere una *distribución a posteriori
sobre funciones*, lo que permite cuantificar analíticamente la
incertidumbre. Esta propiedad es crítica en optimización bayesiana para
equilibrar la exploración y la explotación [@rasmussen2006gpml].

### Definición y consistencia

Un proceso gaussiano (GP) se define formalmente como una colección de
variables aleatorias indexadas por una variable de entrada
$x \in \mathcal{X}$, tal que cualquier subconjunto finito de ellas sigue
una distribución normal multivariante conjunta.

Formalmente, un proceso estocástico $f(x)$ es un GP si, para cualquier
conjunto finito de entradas $X = \{x_1, \dots, x_n\}$, el vector de
evaluaciones $f = [f(x_1), \dots, f(x_n)]^\top$ satisface:
$$f \sim \mathcal{N}(m, K)$$ donde $m$ es el vector de medias y $K$ la
matriz de covarianzas. El proceso queda totalmente especificado por su
función de media $m(x)$ y su función de covarianza $k(x, x')$:
$$m(x) = \mathbb{E}[f(x)]$$
$$k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]$$

Se denota típicamente como $f(x) \sim \mathcal{GP}(m(x), k(x, x'))$. En
la práctica, habitualmente se asume una media nula a priori
($m(x) = 0$), delegando toda la capacidad de modelado a la estructura de
covarianza [@rasmussen2006gpml].

Una propiedad matemática fundamental que hace viables a los GPs es la
*consistencia por marginalización*. Si el modelo especifica una
distribución conjunta para
$(f(x_1), f(x_2)) \sim \mathcal{N}(\mu, \Sigma)$, entonces la
distribución marginal de $f(x_1)$ debe ser necesariamente
$\mathcal{N}(\mu_1, \Sigma_{11})$, donde $\Sigma_{11}$ es el subbloque
correspondiente de la matriz original. Esta coherencia es crucial: nos
permite realizar inferencia probabilística exacta utilizando únicamente
nuestro conjunto finito de datos observados, sin necesidad de calcular o
preocuparnos por los infinitos puntos no observados del dominio.

### Del espacio de pesos al espacio de funciones

Para comprender el origen del kernel de forma rigurosa y entender qué
significa realmente "entrenar" un Proceso Gaussiano, es instructivo
derivar el GP a partir de la regresión lineal bayesiana clásica,
observando cómo se transita de un modelo basado en parámetros a uno
basado en datos y covarianzas.

**El modelo paramétrico y el prior bayesiano**\
Consideremos un modelo estándar proyectado sobre un espacio de
características mediante un vector de funciones base
$\phi(x) \in \mathbb{R}^N$: $$f(x) = \phi(x)^\top w$$ Bajo un paradigma
clásico, la optimización buscaría un único vector de pesos $w$ óptimo.
Sin embargo, desde la perspectiva bayesiana (*weight-space view*),
asumimos que existe incertidumbre sobre cuál es el verdadero valor de
$w$, asignándole una distribución de probabilidad a priori. Típicamente,
se define una distribución normal multivariante con media cero y matriz
de covarianza $\Sigma_p$: $$w \sim \mathcal{N}(0, \Sigma_p)$$

**Marginalización de los pesos**\
En lugar de estimar un valor concreto para $w$, el enfoque bayesiano
requiere considerar todos los modelos posibles promediados por su
probabilidad. Matemáticamente, esto implica integrar (marginalizar) la
variable latente $w$ sobre toda su distribución. Al realizar esta
integral, los parámetros $w$ desaparecen de la formulación
probabilística.

Dado que la combinación lineal de variables con distribución gaussiana
resulta en otra variable gaussiana, la distribución marginal resultante
para la función evaluada en cualquier conjunto de puntos
(*function-space view*) sigue siendo estrictamente normal.

**Transferencia de incertidumbre a las covarianzas**\
Puesto que $f(x)$ conforma una distribución gaussiana, queda
completamente caracterizada por sus dos primeros momentos (media y
covarianza). Analizándolos, obtenemos:

-   **Media:**
    $\mathbb{E}[f(x)] = \mathbb{E}[\phi(x)^\top w] = \phi(x)^\top \mathbb{E}[w] = 0$
    (dado que el prior establece $\mathbb{E}[w] = 0$).

-   **Covarianza:** La covarianza entre las predicciones en dos puntos
    arbitrarios $x$ y $x'$ evalúa cómo varían conjuntamente:
    $$\text{cov}(f(x), f(x')) = \mathbb{E}[\phi(x)^\top w w^\top \phi(x')] = \phi(x)^\top \mathbb{E}[w w^\top] \phi(x') = \phi(x)^\top \Sigma_p \phi(x')$$
    donde se ha aplicado que $\mathbb{E}[w w^\top] = \Sigma_p$.

**El Kernel Trick**\
Es en este último paso donde radica el salto conceptual fundamental del
Proceso Gaussiano: la incertidumbre originalmente modelada en los pesos
del modelo ($\Sigma_p$) se ha transferido íntegramente a la relación
geométrica y espacial entre los puntos de entrada
($\phi(x)^\top \Sigma_p \phi(x')$).

Este resultado motiva de forma natural la definición de la función de
covarianza o *kernel*: $$k(x, x') = \phi(x)^\top \Sigma_p \phi(x')$$ La
inmensa potencia del GP reside en el denominado *kernel trick*. Nos
exime de la necesidad de diseñar explícitamente las funciones base
$\phi(x)$ o de operar con la matriz paramétrica $\Sigma_p$. En su lugar,
podemos definir analíticamente una función $k(x, x')$ que calcule de
forma exacta el resultado escalar de dicha covarianza.

Esta formulación permite trabajar implícitamente con espacios de
características de dimensión infinita (como ocurre al emplear el kernel
exponencial cuadrático o RBF), manteniendo un coste computacional que
depende exclusivamente del número de observaciones $n$, y no de la
dimensionalidad o complejidad de las características internas
[@rasmussen2006gpml].

### Funciones de covarianza (Kernels) {#subsec:gp_kernel}

El kernel encapsula las suposiciones a priori sobre la suavidad y
estructura de la función. Debe cumplir la condición de ser semidefinido
positivo. Los utilizados en este trabajo son:

-   **Lineal (Dot Product):**
    $k(\bm{x}, \bm{x}') = \sigma_0^2 + \bm{x}^\top \bm{x}'$. Corresponde
    a la regresión lineal bayesiana estándar y es útil para capturar
    tendencias globales simples.

-   **Squared Exponential (SE/RBF):** Es el kernel más común por
    defecto.
    $$k_{\mathrm{SE}}(\bm{x},\bm{x}') = \sigma_f^2 \exp\left(-\frac{1}{2}(\bm{x}-\bm{x}')^\top \bm{M} (\bm{x}-\bm{x}')\right).$$
    Produce funciones infinitamente diferenciables, induciendo modelos
    extremadamente suaves.

-   **Matérn:** Introduce un parámetro $\nu$ para controlar la
    diferenciabilidad. donde $r=\|\bm{x}-\bm{x}'\|$ y $\ell$ es la
    longitud característica: $$k_{\text{Mat\'ern}}(r)=
        \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left(\frac{\sqrt{2\nu}\, r}{\ell}\right)^\nu
        K_\nu\left(\frac{\sqrt{2\nu}\, r}{\ell}\right).$$

    Con $\nu=3/2$ (una vez diferenciable) o $\nu=5/2$ (dos veces
    diferenciable), es ideal para modelar fenómenos físicos realistas
    donde el kernel SE resulta demasiado suave.

-   **Ruido Blanco (White Kernel):**
    $k_{\mathrm{White}}(\bm{x}, \bm{x}') = \sigma_n^2 \delta_{\bm{x}, \bm{x}'}$.
    Modela la incertidumbre aleatoria inherente a las observaciones
    (ruido) que no tiene correlación espacial.

##### Isotropía y Automatic Relevance Determination (ARD).

Un kernel se dice *isotrópico* si la correlación depende solo de la
distancia euclídea $\|\bm{x}-\bm{x}'\|$, asumiendo que la función varía
de la misma forma en todas las direcciones. Sin embargo, en problemas de
ingeniería, es habitual que algunas variables de entrada influyan mucho
más que otras en la respuesta.

Para capturar esto, utilizamos **ARD** definiendo la matriz de escalas
$\bm{M} = \mathrm{diag}(\ell_1^{-2}, \dots, \ell_d^{-2})$. Los
hiperparámetros $\ell_d$ son las **escalas de longitud características**
(length-scales), que admiten una interpretación física directa:

-   Si $\ell_d$ es pequeña, la función varía rápidamente en esa
    dimensión (variable muy relevante).

-   Si $\ell_d$ es muy grande ($\ell_d \to \infty$), la función es casi
    constante en esa dirección (variable irrelevante), eliminando
    efectivamente la dependencia.

### Predicción e incertidumbre {#subsec:gp_prediccion}

Dado un conjunto de observaciones ruidosas
$\mathcal{D}_n = \{(\bm{x}_i, y_i)\}_{i=1}^n$ bajo el modelo
$y_i = f(\bm{x}_i) + \varepsilon_i$ con
$\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$, la distribución conjunta
de las observaciones $\bm{y}$ y el valor latente $f_*$ en un punto de
prueba $\bm{x}_*$ es:
$$\begin{bmatrix} \bm{y} \\ f_* \end{bmatrix} \sim \mathcal{N}\left( \bm{0}, \begin{bmatrix} \bm{K} + \sigma_n^2\bm{I} & \bm{k}_* \\ \bm{k}_*^\top & k(\bm{x}_*, \bm{x}_*) \end{bmatrix} \right),$$
En lo que sigue asumimos $m(\bm{x})=0$ (sin pérdida de generalidad en
muchos casos), de modo que las distribuciones a priori y predictivas
tienen media nula, donde $\bm{K}=K(X,X)$ y donde
$\bm{k}_* = K(X,\bm{x}_*)\in\mathbb{R}^{n}$ y
$k_{**}=k(\bm{x}_*,\bm{x}_*)$. Aplicando las reglas de condicionamiento
gaussiano, obtenemos la distribución posterior predictiva
$p(f_* | X, \bm{y}, \bm{x}_*) \sim \mathcal{N}(\bar{f}_*, \mathbb{V}[f_*])$,
dada por: $$\begin{aligned}
\bar{f}_* &= \bm{k}_*^\top (\bm{K} + \sigma_n^2\bm{I})^{-1} \bm{y}, \label{eq:gp_mean} \\
\mathbb{V}[f_*] &= k(\bm{x}_*, \bm{x}_*) - \bm{k}_*^\top (\bm{K} + \sigma_n^2\bm{I})^{-1} \bm{k}_*. \label{eq:gp_var}
\end{aligned}$$

La ecuación ([\[eq:gp_mean\]](#eq:gp_mean){reference-type="ref"
reference="eq:gp_mean"}) representa el predictor promedio. Sin embargo,
la ecuación ([\[eq:gp_var\]](#eq:gp_var){reference-type="ref"
reference="eq:gp_var"}) es el componente crítico para la optimización
global: cuantifica la **incertidumbre** (falta de conocimiento) del
modelo en $\bm{x}_*$. A diferencia de la varianza del ruido
$\sigma_n^2$, este término disminuye a medida que añadimos observaciones
cercanas a $\bm{x}_*$, permitiendo distinguir entre zonas exploradas y
no exploradas.

### Entrenamiento: El Paradigma Bayesiano

En los modelos paramétricos clásicos, el "entrenamiento" consiste en
encontrar el valor exacto de los parámetros internos del modelo (los
pesos $w$) que mejor se ajustan a los datos. En un Proceso Gaussiano,
dichos parámetros $w$ han sido integrados y eliminados de la formulación
probabilística. Por tanto, el entrenamiento de un GP consiste
exclusivamente en inferir los hiperparámetros del kernel, denotados
típicamente como $\theta$ (por ejemplo, la longitud de escala $l$, la
varianza de la señal $\sigma_f^2$ o la varianza del ruido $\sigma_n^2$).

Aquí radica una diferencia fundamental entre ambos enfoques de modelado
[@rasmussen2006gpml]:

-   **Paradigma No-Bayesiano (Frecuentista):** Se busca un único
    conjunto óptimo de parámetros $w$ minimizando una métrica de error
    empírico sobre el conjunto de entrenamiento. Para evitar el
    sobreajuste (*overfitting*), se ven obligados a sumar un término de
    regularización manual a la función de pérdida (por ejemplo, en
    regresión Ridge: $\min \|y - Xw\|^2 + \lambda \|w\|^2$). El peso de
    esta penalización, $\lambda$, debe buscarse empíricamente mediante
    validación cruzada.

-   **Paradigma Bayesiano:** Al asumir que los pesos $w$ son variables
    aleatorias y marginalizarlos, consideramos simultáneamente las
    predicciones de *todos* los modelos posibles, ponderados por su
    probabilidad a priori. Ya no optimizamos funciones sobre un espacio
    de pesos, sino sobre un espacio de hiperparámetros $\theta$ que
    definen las reglas generales (suavidad, amplitud, ruido) del
    universo de funciones resultante.

**La Log-Verosimilitud Marginal (Log Marginal Likelihood)**\
Para encontrar el conjunto óptimo de hiperparámetros $\theta$, el GP
maximiza la probabilidad marginal de haber observado los datos de
entrenamiento $y$, dadas las entradas $X$ y condicionado a $\theta$.
Esta función objetivo se conoce como *Log Marginal Likelihood* (LML):

$$\log p(y|X, \theta) = \underbrace{-\frac{1}{2}y^\top K_y^{-1}y}_{\text{Ajuste a los datos}} \quad \underbrace{-\frac{1}{2}\log|K_y|}_{\text{Penalización por complejidad}} \quad \underbrace{-\frac{n}{2}\log 2\pi}_{\text{Constante}}
\label{eq:lml}$$

donde $K_y = K + \sigma_n^2 I$ es la matriz de covarianza de las
observaciones ruidosas. La belleza analítica de esta ecuación reside en
cómo sus tres componentes compiten de forma natural para guiar el
aprendizaje:

1.  **Término de ajuste a los datos ($-\frac{1}{2}y^\top K_y^{-1}y$):**
    Representa la distancia de Mahalanobis entre las observaciones y la
    media del modelo (asumida cero). Evalúa cómo de bien encajan los
    datos reales con la estructura de covarianza impuesta por el kernel.
    Penaliza fuertemente a los modelos que no son capaces de pasar cerca
    de las observaciones.

2.  **Término de penalización por complejidad
    ($-\frac{1}{2}\log|K_y|$):** El determinante de la matriz de
    covarianza, $|K_y|$, actúa como una medida del "volumen" o
    flexibilidad del modelo. Un kernel con hiperparámetros que permitan
    una variabilidad extrema (por ejemplo, una longitud de escala $l$
    minúscula) generará un determinante inmenso. Al estar restando, esto
    hundirá la probabilidad del modelo. Es la consecuencia directa de
    que la probabilidad marginal debe sumar uno: si un modelo es capaz
    de explicar cualquier patrón de datos imaginable, la masa de
    probabilidad que le asigna específicamente a nuestro conjunto $y$
    observado se diluye drásticamente.

3.  **Constante de normalización ($-\frac{n}{2}\log 2\pi$):** Depende
    únicamente del número de datos $n$ y garantiza que la función sea
    una distribución de probabilidad válida, aunque no afecta a los
    gradientes durante la optimización de $\theta$.

**La Navaja de Ockham Automática**\
La maximización de la Ecuación
[\[eq:lml\]](#eq:lml){reference-type="ref" reference="eq:lml"} incorpora
de forma intrínseca el principio de la Navaja de Ockham
[@rasmussen2006gpml]. En el paradigma bayesiano, no es necesario añadir
parámetros de regularización externos ni recurrir a costosas
validaciones cruzadas. El equilibrio o "tensión" entre el término de
ajuste a los datos (que favorece modelos complejos para reducir el
error) y el término del determinante (que penaliza la excesiva
flexibilidad) empuja sistemáticamente la optimización hacia una solución
de compromiso. El resultado es la selección automática del modelo más
simple que sea capaz de explicar los datos adecuadamente.

## Diseño experimental y muestreo inicial

### LHS, SOBOL y muestreo espacio-relleno {#sec:muestreo_inicial}

El éxito de la optimización basada en modelos sustitutos (SBO) depende
de la calidad del conjunto de datos inicial $\mathcal{D}_n$. Si las
observaciones iniciales se concentran en una región muy reducida del
espacio de búsqueda $\mathcal{X} \subset \mathbb{R}^d$, el modelo
sufrirá de una falta de información severa. En el caso de un Proceso
Gaussiano, esto se traduce en una varianza predictiva (incertidumbre
epistémica) artificialmente baja en la zona densa y arbitrariamente alta
en el resto del dominio, perjudicando el rendimiento de las funciones de
adquisición posteriores. Por ello, es fundamental emplear técnicas de
Diseño de Experimentos (DoE) que garanticen una exploración inicial
representativa.

**El problema del Muestreo Aleatorio**\
El enfoque más elemental es el muestreo aleatorio uniforme (Monte Carlo
estándar). Sin embargo, en dimensiones moderadas o altas, la
aleatoriedad pura tiende a generar clústeres y grandes vacíos
informativos. Esto resulta ineficiente cuando el presupuesto máximo de
evaluaciones $B$ es estricto y el coste de evaluación $c(x)$ es elevado.

**Muestreo Espacio-Relleno (*Space-Filling*)**\
Para mitigar las deficiencias del muestreo aleatorio en problemas
continuos es habitual usar muestreos *space-filling*, como Latin
Hypercube Sampling (LHS) o SOBOL [@forrester2008engineering]. Su
objetivo matemático es distribuir los $n$ puntos iniciales de forma que
se maximice la distancia entre ellos, cubriendo el dominio de la manera
más uniforme posible antes de iniciar la búsqueda adaptativa secuencial.

**Latin Hypercube Sampling (LHS)**\
El Muestreo de Hipercubo Latino (LHS) es una de las técnicas
*space-filling* más consolidadas. Para generar $n$ puntos en $d$
dimensiones, el rango de cada variable se divide en $n$ intervalos de
igual probabilidad marginal. Posteriormente, se selecciona un punto en
cada intervalo de forma que, al proyectar el conjunto de datos sobre
cualquier eje dimensional, exista exactamente un punto por intervalo
(análogo a la regla de no repetición de un tablero de Sudoku).

Aunque garantiza una estratificación marginal perfecta, un LHS estándar
no asegura una buena cobertura multidimensional geométrica (por puro
azar, los puntos podrían alinearse en la diagonal del hipercubo). En la
práctica, se suelen emplear variantes como *Maximin LHS*, que generan
múltiples diseños aleatorios y seleccionan aquel que maximiza la
distancia mínima entre cualquier par de puntos.

**Secuencias de Sobol (Quasi-Monte Carlo)**\
Las secuencias de Sobol pertenecen a la familia de métodos Quasi-Monte
Carlo y generan sucesiones de baja discrepancia (*low-discrepancy
sequences*). A diferencia del muestreo aleatorio puro o del LHS, las
secuencias de Sobol son estrictamente deterministas. Utilizan fracciones
en base 2 y polinomios primitivos sobre el campo de Galois para
subdividir el espacio iterativamente, rellenando los vacíos dejados por
los puntos anteriores de una forma geométricamente óptima.

Esta técnica presenta dos ventajas cruciales para inicializar modelos
sustitutos:

1.  **Cobertura superior:** Evita de forma matemática la formación de
    clústeres, proporcionando una cobertura del espacio multidimensional
    más uniforme. Esto garantiza al modelo probabilístico (como el GP)
    una cuadrícula de información estable, logrando que la varianza
    predictiva inicial esté mejor acotada en todo el dominio.

2.  **Determinismo y reproducibilidad:** Al no depender de una semilla
    aleatoria (*seed*), los experimentos iniciales son completamente
    reproducibles sin varianza intrínseca, lo cual facilita la
    evaluación objetiva y la depuración de algoritmos.

**Consideraciones prácticas: El tamaño del diseño inicial**\
Independientemente de la técnica de muestreo, una heurística ampliamente
aceptada en la literatura de optimización global (conocida como la regla
empírica de Jones) establece que el tamaño del muestreo inicial debe ser
proporcional a la dimensionalidad del problema. Típicamente, se
recomienda un tamaño $n = 10d$, donde $d$ es la dimensión del vector de
entradas [@jones1998efficient]. En escenarios donde las evaluaciones
imponen un coste extremo, este presupuesto inicial suele reducirse a
$n = 5d$ para reservar un mayor número de iteraciones para la fase de
optimización guiada por el modelo.

## Selección de nuevas evaluaciones

Una vez construido un modelo sustituto inicial a partir del diseño de
experimentos, surge el reto de cómo utilizarlo para encontrar el óptimo
global del problema original. En la literatura existen tres enfoques
principales para abordar esta tarea [@jiang2020surrogate]:

1.  **Heurísticas puras:** Algoritmos como los genéticos o evolutivos
    que evalúan directamente la función de caja negra. Requieren miles
    de evaluaciones de alta fidelidad, lo cual resulta inviable bajo
    presupuestos estrictos.

2.  **Metaheurísticas asistidas por sustitutos:** Utilizan el modelo
    sustituto para pre-evaluar y filtrar individuos dentro de una
    población evolutiva antes de ejecutar la simulación real. Aunque
    reducen el coste, su naturaleza poblacional sigue demandando un
    número considerable de muestras.

3.  **Optimización Basada en Modelos Sustitutos (SBO):** Es el enfoque
    más eficiente para problemas costosos. Se apoya en un modelo
    probabilístico para guiar la búsqueda secuencialmente, evaluando un
    único punto (o un pequeño lote) en cada iteración, maximizando la
    información obtenida.

### Planteamiento: optimización secuencial con presupuesto limitado

El flujo de trabajo en SBO, y particularmente en Optimización Bayesiana
(BO), guarda una estrecha relación con paradigmas de aprendizaje
automático como el *Active Learning*. Partiendo de un conjunto de datos
inicial $\mathcal{D}_n$, el algoritmo no busca optimizar la función
objetivo original directamente, sino que optimiza una *función de
adquisición* o *infill criterion*.

Esta función, mucho más barata de evaluar, decide de forma inteligente
cuál es el candidato óptimo $x^*$ para la siguiente iteración. Una vez
determinado, se evalúa en el sistema real (costoso), se añade a
$\mathcal{D}_n$, y se reentrena el modelo sustituto (actualizando sus
hiperparámetros). Este ciclo se repite hasta agotar el presupuesto de
evaluaciones $B$ o hasta satisfacer un criterio de convergencia.

### Exploración vs explotación

El diseño de la función de adquisición define el comportamiento del
algoritmo, debiendo equilibrar dos fuerzas contrapuestas
[@jiang2020surrogate]:

-   **Explotación (Búsqueda local):** Priorizar las regiones del espacio
    de búsqueda donde el modelo sustituto predice que el valor de la
    función objetivo es muy bueno (mínimo, en el caso de minimización).
    Una estrategia puramente explotadora converge rápido, pero es
    altamente susceptible a quedar atrapada en mínimos locales.

-   **Exploración (Búsqueda global):** Priorizar regiones con alta
    incertidumbre, es decir, donde el modelo tiene pocos datos y la
    varianza predictiva es alta. Una estrategia puramente exploradora
    actúa como un muestreo pseudoaleatorio, desperdiciando evaluaciones
    en zonas poco prometedoras.

Gracias al uso de Procesos Gaussianos, disponemos analíticamente tanto
de la predicción puntual $\mu(x)$ como de la incertidumbre $\sigma(x)$,
permitiendo formular funciones de adquisición balanceadas.

### Optimización Bayesiana y funciones de adquisición

Aunque existen diversos criterios (como *Probability of Improvement* o
*Lower Confidence Bound*), en este trabajo nos centraremos en el
*Expected Improvement*, considerado el estándar *de facto* por su
robustez matemática.

**Probability of Improvement (PI)**\
El criterio PI fue uno de los primeros desarrollados. Su objetivo es
maximizar la probabilidad de que un nuevo punto $x$ mejore la mejor
solución observada hasta el momento, $f_{\min}$. Si asumimos que la
predicción sigue una distribución normal
$\mathcal{N}(\mu(x), \sigma^2(x))$, la probabilidad de mejora es:
$$PI(x) = P(Y(x) < f_{\min}) = \Phi\left(\frac{f_{\min} - \mu(x)}{\sigma(x)}\right)$$
donde $\Phi$ es la función de distribución acumulada de la normal
estándar. La limitación principal de PI es que premia cualquier mejora
por minúscula que sea, sesgando fuertemente el algoritmo hacia una
explotación local excesiva.

**Expected Improvement (EI)**\
El *Expected Improvement* (Mejora Esperada) resuelve la deficiencia de
PI considerando no solo *si* un punto mejorará el óptimo actual, sino
*cuánto* lo mejorará [@jones1998efficient]. Definimos la función de
mejora $I(x)$ frente a la mejor observación actual $f_{\min}$ como:
$$I(x) = \max(0, f_{\min} - Y(x))$$ Dado que $Y(x)$ es una variable
aleatoria modelada por el GP, podemos calcular el valor esperado de esta
mejora integrando sobre su función de densidad de probabilidad. Tras el
desarrollo analítico, la formulación cerrada de EI resulta en:
$$E[I(x)] = \begin{cases} 
(f_{\min} - \mu(x)) \Phi(Z) + \sigma(x) \phi(Z) & \text{si } \sigma(x) > 0 \\
0 & \text{si } \sigma(x) = 0 
\end{cases}
\label{eq:ei}$$ donde $Z = \frac{f_{\min} - \mu(x)}{\sigma(x)}$, y
$\phi(\cdot)$ representa la función de densidad de probabilidad de la
normal estándar.

La Ecuación [\[eq:ei\]](#eq:ei){reference-type="ref" reference="eq:ei"}
es una síntesis perfecta y elegante de las dos estrategias de búsqueda:

-   El primer término, $(f_{\min} - \mu(x)) \Phi(Z)$, domina cuando
    $\mu(x)$ es menor que $f_{\min}$, impulsando la **explotación** de
    zonas prometedoras.

-   El segundo término, $\sigma(x) \phi(Z)$, domina cuando la
    incertidumbre $\sigma(x)$ es grande, forzando la **exploración** de
    regiones desconocidas.

Matemáticamente, esta síntesis de estrategias se demuestra de forma
analítica calculando las derivadas parciales de la función EI respecto a
los momentos predictivos del modelo.

Por un lado, la derivada respecto a la media predictiva es:
$$\frac{\partial E[I(x)]}{\partial \mu(x)} = -\Phi(Z)$$ Dado que la
función de distribución acumulada $\Phi(Z)$ toma valores estrictamente
positivos, esta derivada es siempre negativa. Esto demuestra que la
función EI es **monótonamente decreciente** respecto a $\mu(x)$. En la
práctica, significa que cuanto menor (y por tanto, mejor en un problema
de minimización) sea el valor predicho por el modelo para una región,
mayor será su expectativa de mejora, lo que empuja al algoritmo a
*explotar* las zonas prometedoras.

Por otro lado, la derivada respecto a la incertidumbre predictiva
resulta en: $$\frac{\partial E[I(x)]}{\partial \sigma(x)} = \phi(Z)$$
Puesto que la función de densidad de la normal estándar $\phi(Z)$ es
siempre mayor que cero, esta derivada es estrictamente positiva. Por
consiguiente, EI es **monótonamente creciente** respecto a $\sigma(x)$.
Su significado práctico es que, ante dos localizaciones con la misma
predicción media, el algoritmo siempre asignará un mayor valor de
adquisición a la que posea mayor incertidumbre, formalizando así la
*exploración*. Un mayor desconocimiento amplía la varianza de la campana
de Gauss predictiva, aumentando el área de probabilidad que cae por
debajo del umbral $f_{\min}$.

En los puntos de diseño ya observados del conjunto de datos (asumiendo
observaciones sin ruido), la certidumbre del Proceso Gaussiano es
absoluta, es decir, $\sigma(x) = 0$. Al evaluar este límite, la mejora
esperada se anula ($E[I(x)] = 0$). Esta propiedad es crítica, ya que
previene de forma inherente el estancamiento del algoritmo, garantizando
que no se desperdicie el valioso presupuesto computacional evaluando
exactamente el mismo punto dos veces.

Una vez que este ciclo secuencial agota el presupuesto máximo de
evaluaciones $B$, o se alcanza un criterio de convergencia predefinido,
el proceso de enriquecimiento (*infill*) se da por terminado. El
resultado final es una estimación del óptimo global y un modelo
sustituto refinado que ha aprovechado al máximo el equilibrio entre la
exploración y la explotación del espacio de búsqueda. De este modo, la
Optimización Bayesiana garantiza una búsqueda matemáticamente eficiente
y efectiva, logrando el mejor rendimiento posible bajo restricciones
computacionales severas.

## Métricas de evaluación

Para validar el rendimiento de los modelos sustitutos, no es suficiente
evaluar la precisión de sus predicciones puntuales. En el caso de
modelos probabilísticos como los Procesos Gaussianos, resulta
fundamental cuantificar la calidad de su incertidumbre predictiva. Por
ello, las métricas de evaluación se dividen en dos categorías: métricas
predictivas clásicas y métricas probabilísticas.

### Métricas predictivas (RMSE, MAE, R$^2$)

Estas métricas evalúan la divergencia entre la observación real $y_i$ y
la media predictiva del modelo $\mu_i$ (o $\hat{y}_i$) sobre un conjunto
de prueba (*test set*) de tamaño $N$.

-   **RMSE (Root Mean Squared Error):** Es la métrica estándar para
    evaluar el error de regresión. Al elevar las diferencias al
    cuadrado, penaliza de forma mucho más severa los errores grandes
    (valores atípicos o predicciones desastrosas) en comparación con los
    errores pequeños.
    $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \mu_i)^2}$$

-   **MAE (Mean Absolute Error):** Representa la magnitud promedio de
    los errores en las predicciones, sin considerar su dirección. Crece
    de forma lineal, lo que lo hace más robusto frente a valores
    atípicos que el RMSE.
    $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \mu_i|$$

-   **Coeficiente de determinación ($R^2$):** Mide la proporción de la
    varianza total de la variable dependiente que es explicada por el
    modelo. Un $R^2$ de 1 indica un ajuste perfecto, mientras que un
    $R^2$ de 0 (o negativo) indica que el modelo no es mejor que
    predecir siempre la media empírica de los datos $\bar{y}$.
    $$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \mu_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$

### Métricas probabilísticas (NLL, NLPD, Cobertura del 95%)

Mientras que las métricas predictivas ignoran la varianza $\sigma_i^2$
devuelta por el Proceso Gaussiano, las métricas probabilísticas evalúan
la calibración conjunta de la media y la incertidumbre. El objetivo es
comprobar si el modelo \"sabe lo que no sabe\".

**Negative Log-Likelihood (NLL) y Negative Log Predictive Density
(NLPD)**\
En la literatura de *Machine Learning*, NLL y NLPD a veces se utilizan
de forma intercambiable, pero conceptualmente cumplen roles distintos
según la fase del modelado.

El **NLL** se emplea típicamente como la función objetivo durante la
fase de **entrenamiento**. Equivale matemáticamente a la
Log-Verosimilitud Marginal (Ecuación
[\[eq:lml\]](#eq:lml){reference-type="ref" reference="eq:lml"})
multiplicada por $-1$. El algoritmo de optimización busca minimizar este
valor evaluando sobre todo el conjunto de entrenamiento, ajustando así
los hiperparámetros del modelo.

Por el contrario, en la fase de **evaluación**, la métrica adecuada para
medir el rendimiento probabilístico sobre un conjunto de prueba (*test
set*) con datos no vistos es el **NLPD**. El NLPD evalúa punto a punto
la probabilidad de observar el valor real $y_i$ dada la distribución
predictiva marginal $\mathcal{N}(\mu_i, \sigma_i^2)$ que el modelo ya
entrenado genera para esa nueva entrada.

Para un conjunto de prueba de $N$ muestras, asumiendo predicciones
gaussianas independientes, la métrica NLPD se define como:

$$ \text{NLPD} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{2} \log(2\pi\sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2} \right)
\label{eq:nlpd} $$

Un modelo mejor calibrado devolverá valores más bajos de NLPD.
Analizando la Ecuación [\[eq:nlpd\]](#eq:nlpd){reference-type="ref"
reference="eq:nlpd"}, se observa que penaliza drásticamente dos
escenarios indeseables:

1.  **Sobreconfianza (*Overconfidence*):** Si el modelo está muy seguro
    de su predicción ($\sigma_i^2 \approx 0$) pero comete un error
    grande alejándose del valor real $y_i$, el segundo término de la
    ecuación se disparará, resultando en un NLPD enorme.

2.  **Subconfianza (*Underconfidence*):** Si el modelo devuelve una
    incertidumbre muy alta en todo el dominio para asegurarse de cubrir
    siempre el valor real, el primer término (que depende del logaritmo
    de la varianza) crecerá de forma continua, penalizando la falta de
    capacidad informativa y utilidad práctica del modelo.

**Cobertura de Intervalos (Coverage 95%)**\
La Cobertura del 95% (Cov$_{95}$) es una métrica empírica directa para
evaluar la calibración del modelo. Consiste en calcular qué porcentaje
de las observaciones reales del conjunto de prueba caen dentro del
intervalo de confianza del 95% predicho por el modelo.

Sabiendo que para una distribución normal el 95% de la masa de
probabilidad se concentra a $1.96$ desviaciones típicas de la media,
definimos el intervalo predictivo como
$I_i = [\mu_i - 1.96\sigma_i, \mu_i + 1.96\sigma_i]$. La métrica se
calcula mediante una función indicatriz $\mathbb{I}$:

$$\text{Cov}_{95} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_i \in I_i)$$

Un modelo perfectamente calibrado debería obtener un
Cov$_{95} \approx 0.95$.

-   Si el valor es considerablemente **menor** a 0.95 (ej. 0.60),
    significa que las observaciones reales están cayendo fuera de las
    bandas de incertidumbre: el modelo está pecando de
    **sobreconfianza**.

-   Si el valor es de 1.0, el modelo está conteniendo todos los puntos,
    lo cual suele indicar bandas de incertidumbre excesivamente anchas e
    inútiles (**subconfianza**).
