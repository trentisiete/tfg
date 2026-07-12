# Anotación 11: justificar por qué no se usa Tenebrio

## Anotación o motivo

La frase actual explica primero que los datos de \textit{Tenebrio} no son inválidos, pero eso no ayuda a entender la decisión. Conviene decir directamente por qué no se han usado.

## Ubicación

Archivo de trabajo: `TFG_José/working_on/cap3.tex`.

Sección: `\subsection{Alcance del caso real y selección de \textit{Hermetia}}`.

## Rango de sustitución

Sustituir desde:

```latex
Esta decisión no implica que los datos de \textit{Tenebrio} sean inválidos. La razón es de alcance experimental y homogeneidad del caso modelado. El bloque de \textit{Hermetia} utilizado en el pipeline contiene 33 observaciones, organizadas en 11 dietas con 3 réplicas por dieta, todas pertenecientes al mismo bloque de estudio. Además, las covariables necesarias para los dos modos de entrada están completas y los objetivos presentan una disponibilidad suficientemente homogénea: \texttt{FCR} cuenta con 31 valores válidos y \texttt{QUITINA (\%)} y \texttt{PROTEINA (\%)} cuentan con 33 valores.
```

hasta:

```latex
En cambio, aunque \textit{Tenebrio} también se genera como dataset procesado, mezcla varios bloques experimentales y tipos adicionales de dieta, como los asociados a posos de café y orujillo. Estos datos incorporan diferencias de tratamiento, especialmente en la forma de suministrar agua en algunos controles, y presentan una disponibilidad distinta de objetivos. Por ello, \textit{Tenebrio} se reserva como posible segundo caso o subanálisis posterior, pero no se mezcla con \textit{Hermetia} en los resultados actuales.
```

## Texto propuesto

```latex
Sin embargo, los datos de \textit{Tenebrio molitor} no se han utilizado porque mezclan varios bloques experimentales, incorporan tipos adicionales de dieta y presentan diferencias de tratamiento, como el suministro de agua en algunos controles. Para mantener un caso cerrado y comparable, el análisis se limita a \textit{Hermetia illucens}: 33 observaciones organizadas en 11 dietas con 3 réplicas por dieta. En este bloque, las variables de entrada están completas y los objetivos tienen disponibilidad suficiente: FCR cuenta con 31 valores válidos, mientras que quitina y proteína cuentan con 33.
```

## Relación con otras partes

- Simplifica la justificación y evita una defensa innecesaria de \textit{Tenebrio}.
- Refuerza el criterio metodológico: homogeneidad del bloque analizado.

## Notas fuera del LaTeX

La redacción evita nombres internos de archivos y mantiene los números necesarios para justificar el alcance.
