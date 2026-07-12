# Anotación 20: aclarar y regenerar la figura de error por dieta

## Anotación o motivo

La figura mezcla dos magnitudes distintas: el color representa el error normalizado por el rango de cada objetivo, mientras que el número escrito en cada celda muestra el MAE medio en las unidades originales. Por eso la barra llega aproximadamente a \(0.4\), pero algunas celdas muestran valores superiores a \(5\). No es un error de cálculo, sino un problema de lectura visual.

Además, FCR no se expresa en porcentaje: es una razón de conversión alimento/biomasa. En cambio, quitina y proteína sí están en porcentaje. Esta diferencia debe quedar explícita para no sugerir que todas las celdas comparten la misma unidad.

La mejor solución es regenerar la figura con una leyenda más explícita y un formato más compacto. No conviene etiquetar la barra como "mayor MAE", porque el color no muestra el MAE bruto, sino el error relativo dentro de cada objetivo.

## Verificación realizada

- La figura se genera en `generate_entomotive_pov_figures.py`, función `plot_error_by_diet`.
- El archivo `fig_02_error_by_left_out_diet_data.csv` contiene dos columnas: `abs_error` y `normalized_error`.
- El texto de las celdas procede de `abs_error`.
- El color procede de `normalized_error = abs_error / rango_del_objetivo`.

## Archivo propuesto

Se ha generado una versión revisada para comparar:

`TFG_José/working_on/fig_02_error_by_left_out_diet_propuesta.png`

Esta versión mantiene los mismos datos, pero:

- reduce el título;
- explicita que amarillo es menor error relativo y rojo mayor error relativo;
- aclara debajo de la figura que los números son MAE en unidades originales: FCR sin porcentaje, quitina y proteína en porcentaje;
- mantiene la normalización por objetivo para poder comparar FCR, quitina y proteína en una misma escala visual.

## Ubicación

Archivo de referencia: `TFG_José/chapters/03_metodologia.tex`.

Sección: `\subsubsection{Generalización estructural frente a memorización}`.

Figura afectada: `fig:error_by_left_out_diet`.

## Cambio 1: sustituir el archivo de figura

Sustituir el recurso actual:

```latex
assets/fig_02_error_by_left_out_diet.png
```

por una versión regenerada con la leyenda corregida. Si se acepta la propuesta visual, copiar la imagen propuesta sobre el recurso final con el mismo nombre para no cambiar la referencia LaTeX.

## Cambio 2: ajustar el pie de figura

Sustituir:

```latex
\caption[Error por dieta retenida en LODO]{Dificultad de generalización por dieta retenida en el protocolo LODO. El color representa el error normalizado por el rango del objetivo y el texto muestra el MAE medio en las unidades originales de cada variable.}
```

por:

```latex
\caption[Error por dieta retenida en LODO]{Dificultad de generalización por dieta retenida en el protocolo LODO. El color muestra el error relativo al rango de cada objetivo, de amarillo a rojo según menor o mayor dificultad. El número de cada celda muestra el MAE medio en las unidades originales de la variable correspondiente: FCR como razón sin porcentaje, quitina y proteína en porcentaje.}
```

## Cambio 3: ajustar la frase de interpretación

Sustituir:

```latex
No obstante, esta mejora agregada no debe interpretarse como una generalización uniforme. La Figura~\ref{fig:error_by_left_out_diet} descompone el error por dieta retenida y muestra que la dificultad del problema depende de la formulación evaluada.
```

por:

```latex
No obstante, esta mejora agregada no debe interpretarse como una generalización uniforme. La Figura~\ref{fig:error_by_left_out_diet} descompone el error por dieta retenida: el color permite comparar la dificultad relativa dentro de cada objetivo y los valores anotados conservan el MAE en sus unidades originales.
```

## Cambio recomendado en el script de generación

En `generate_entomotive_pov_figures.py`, dentro de `plot_error_by_diet`, conservar la lógica de cálculo y modificar solo la parte visual de la figura:

```python
fig, ax = plt.subplots(figsize=(6.6, 5.25))
sns.heatmap(
    pivot_norm,
    annot=annot,
    fmt="",
    cmap="YlOrRd",
    vmin=0,
    vmax=0.40,
    linewidths=0.45,
    linecolor="white",
    cbar_kws={
        "label": "Color: error relativo por objetivo\n(amarillo = menor, rojo = mayor)",
        "ticks": [0, 0.10, 0.20, 0.30, 0.40],
        "shrink": 0.82,
    },
    annot_kws={"fontsize": 9},
    ax=ax,
)
ax.set_title("Dificultad LODO por dieta retenida", fontsize=12.5, weight="bold", pad=8)
ax.set_xlabel("Objetivo")
ax.set_ylabel("Dieta retenida")
ax.tick_params(axis="x", labelrotation=0)
ax.tick_params(axis="y", labelrotation=0)
fig.text(
    0.5,
    0.02,
    "Cada celda muestra el MAE medio en unidades originales: FCR sin %, quitina y proteína en %.",
    ha="center",
    va="bottom",
    fontsize=8.5,
)
fig.tight_layout(rect=[0, 0.045, 1, 1])
```

## Relación con otras partes

- La figura sigue apoyando el mismo argumento: la mejora agregada no es uniforme por dieta.
- La normalización por objetivo sigue siendo necesaria, porque FCR, quitina y proteína tienen escalas distintas.
- El texto evita sugerir que la barra de color y los números de las celdas están en la misma unidad.
- FCR debe tratarse como una razón sin porcentaje; no escribir FCR en \% junto a quitina y proteína.

## Notas fuera del LaTeX

La figura propuesta está en `working_on` y no sustituye todavía al recurso final de `assets`.
