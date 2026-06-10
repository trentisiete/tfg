# Guia tecnica de metricas del caso real Entomotive sin TPC

Este documento es una guia tecnica para el agente redactor. No es prosa de TFG y no sustituye a los resultados del pipeline. Describe la variante solicitada del caso real Entomotive/Hermetia sin `TPC_larva_media` como target.

## Alcance operativo

- Caso: Entomotive real, especie modelada `Hermetia`.
- Dataset modelado: `data/entomotive_datasets/productivity_hermetia_lote.csv`.
- Targets activos: `FCR`, `QUITINA (%)` como `Quitina`, `PROTEINA (%)` como `Proteina`.
- Target excluido: `TPC_larva_media` como `TPC`, porque el resultado no es concluyente y se quiere limpiar la salida.
- Nota: `TPC_dieta_media` puede seguir apareciendo como covariable en los feature sets de `src/configs/tuning_specs.py`; lo que queda excluido aqui es TPC como variable objetivo.
- Validacion: LODO por dieta, usando `diet_name` como grupo.
- Modelos activos: `Dummy`, `GP_Linear`, `GP_RBF_NoARD`, `GP_Matern32_NoARD`, `GP_Matern52_NoARD`, `GP_Compuesto_NoARD` y la variante ARD selectiva del mejor kernel previo cuando aplique.
- Modelos prohibidos en este caso: arboles, ensembles, Ridge, PLS u otros regresores.

## Rutas que debe conocer el agente

Codigo:

- `run_exhaustive_tuning.py`: orquesta la ejecucion del caso real, carga `productivity_hermetia_lote.csv`, crea `X`, `y` y `groups`, llama a `nested_lodo_tuning(..., primary="mae")` y guarda JSON/CSV por target, modo de features y modelo.
- `src/analysis/tuning.py`: implementa el LODO anidado. En el bucle interno selecciona hiperparametros; en el bucle externo calcula metricas finales honestas sobre la dieta dejada fuera.
- `src/analysis/surrogate_metrics.py`: implementacion real de MAE, RMSE, R2, NLPD y coverage. No existe `src/metrics/regression.py` en este repositorio.
- `src/models/base.py`: expone `SurrogateRegressor.compute_metrics`, que llama a `compute_surrogate_metrics` cuando `extended=True`.
- `src/models/gp.py`: el GP devuelve media y desviacion estandar con `predict_dist`, necesarias para NLPD y coverage.
- `src/models/dummy.py`: el Dummy devuelve prediccion puntual; no aporta incertidumbre propia, por lo que NLPD y coverage no deben interpretarse para Dummy.
- `src/configs/tuning_specs.py`: define targets, features, modelos, kernels y grids. En la variante sin TPC debe iterar solo por `FCR`, `Quitina` y `Proteina`.
- `tuning_visual_report.py`: consume los summaries/folds del tuning y genera graficas de estabilidad, paridad, incertidumbre GP, perfiles de respuesta y tabla de parametros ajustados.
- `audit_entomotive_real_case.py`: audita dataset, ranking observado, metricas activas y figuras de resumen. En la variante sin TPC debe recalcular ranking observado solo con FCR, Quitina y Proteina.

Artefactos esperados para la variante sin TPC:

- Logs esperados: `outputs/logs/tuning/productivity_hermetia_gp_ard_kernels_no_tpc_v1`.
- Graficas esperadas: `outputs/plots/comprehensive_report_gp_ard_kernels_no_tpc_v1`.
- Reportes esperados: `outputs/reports/entomotive_real_case_no_tpc_v1`.
- Subcarpetas esperadas por target: `fcr`, `quitina`, `proteina`.
- No debe existir una carpeta nueva `tpc` en esos artefactos.

## Flujo donde aparecen las metricas

1. `run_exhaustive_tuning.py` construye los datos con `build_X_y_groups`.
   - Elimina filas donde el target activo sea nulo.
   - Usa `diet_name` como `groups`.
   - Codifica `byproduct_type` con one-hot.
   - Usa `FEATURE_COLS_REDUCED` o `FEATURE_COLS_FULL` desde `src/configs/tuning_specs.py`.

2. `run_exhaustive_tuning.py` llama a `nested_lodo_tuning` con `primary="mae"`.
   - Esta es la metrica primaria de seleccion de hiperparametros.
   - Lower is better: el mejor set de parametros es el que minimiza el MAE medio en el LODO interno.

3. `src/analysis/tuning.py` ejecuta LODO anidado.
   - Outer LODO: deja fuera una dieta completa y evalua el modelo final sobre esa dieta.
   - Inner LODO: dentro del entrenamiento de cada outer fold, deja fuera dietas de entrenamiento para elegir hiperparametros.
   - Cada fold externo tiene como test todas las replicas de una dieta, evitando fuga entre replicas.

4. En cada outer fold, el modelo entrenado predice `mean, std` con `predict_dist`.
   - En GP, `std` viene del `GaussianProcessRegressor`.
   - En Dummy, `std` es `None`.

5. `compute_metrics(..., extended=True)` calcula las metricas finales del fold.
   - Se guardan en los CSV de folds.
   - Se agregan en `summary.macro` y `summary.micro`.

6. `tuning_visual_report.py` carga folds y summaries.
   - Usa MAE, RMSE y coverage95 para boxplots de estabilidad.
   - Usa MAE y R2 en parity plots.
   - Usa `std` GP para graficas de incertidumbre y bandas al 95%.
   - Genera `gp_fitted_parameters.png` con `alpha` seleccionado y `kernel_` optimizado.

7. `audit_entomotive_real_case.py` consolida metricas.
   - Produce `active_model_metrics.csv`.
   - Produce graficas de MAE y coverage para los GP.
   - En la variante sin TPC debe consolidar solo FCR, Quitina y Proteina.

## LODO por dieta

LODO significa Leave-One-Diet-Out en este caso, implementado con `LeaveOneGroupOut` y `groups = diet_name`.

La razon tecnica es evitar fuga de informacion entre replicas. Si se separasen filas al azar, replicas de una misma dieta podrian caer a la vez en train y test. Eso haria que el modelo evaluase casi la misma formulacion que ya ha visto, inflando artificialmente las metricas.

Con LODO, cuando se evalua una dieta:

- Ninguna replica de esa dieta esta en entrenamiento.
- La prediccion simula el caso de una dieta/formulacion observada como grupo completo no visto.
- El numero esperado de folds es el numero de dietas, actualmente 11 para Hermetia.
- FCR tiene menos muestras validas que Quitina/Proteina si hay filas sin FCR; aun asi el grupo de validacion sigue siendo la dieta.

## Metrica primaria: MAE

Formula:

```text
MAE = mean(abs(y_true - y_pred))
```

Interpretacion:

- Mide el error absoluto medio en las unidades del target.
- Es robusta y directa para comparar modelos en un dataset pequeno.
- Cuanto menor, mejor.
- Es la metrica primaria del pipeline real: `primary="mae"`.

Uso exacto:

- Inner LODO en `src/analysis/tuning.py`: selecciona hiperparametros minimizando MAE medio.
- Outer LODO en `src/analysis/tuning.py`: reporta MAE final por dieta dejada fuera.
- Summaries: `summary["macro"]["mae"]` y `summary["micro"]["mae"]`.
- Auditoria: `active_model_metrics.csv` debe incluir `mae_macro_mean`.
- Graficas: `stability_mae.png`, `active_mae_reduced_features.png`, `active_mae_full_features.png` y comparativas full vs reduced.

Criterio de lectura:

- Para decir que un GP aporta valor, debe mejorar MAE frente a Dummy de forma clara para el target y modo de features.
- Si MAE mejora pero coverage/NLPD son malos, la prediccion puntual puede ser util, pero la incertidumbre no debe venderse como bien calibrada.

## RMSE

Formula:

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
```

Interpretacion:

- Penaliza mas los errores grandes que MAE.
- Cuanto menor, mejor.
- Esta en las mismas unidades del target.

Uso exacto:

- Se calcula en `src/analysis/surrogate_metrics.py`.
- Se guarda por fold y se agrega en `summary["macro"]["rmse"]` y `summary["micro"]["rmse"]`.
- `tuning_visual_report.py` genera `stability_rmse.png`.

Criterio de lectura:

- Es diagnostico, no criterio primario.
- Si RMSE se separa mucho de MAE, hay dietas/folds con errores grandes.
- En caso real pequeno, debe usarse para detectar riesgo de fallos puntuales, no para seleccionar hiperparametros.

## R2

Formula:

```text
R2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
```

Interpretacion:

- Mide proporcion de variabilidad explicada frente a predecir la media.
- Puede ser negativa si el modelo es peor que la media.
- Cuanto mayor, mejor, pero es inestable con pocos puntos por fold.

Uso exacto:

- Se calcula en `src/analysis/surrogate_metrics.py` cuando `extended=True`.
- Se guarda por outer fold.
- `tuning_visual_report.py` tambien calcula R2 global en los parity plots usando todas las predicciones CV reconstruidas.

Criterio de lectura:

- Es diagnostico secundario.
- No debe usarse como criterio de seleccion de hiperparametros.
- En LODO con dietas de 2-3 replicas, el R2 por fold puede ser ruidoso; conviene interpretarlo junto al parity plot y MAE.

## NLPD

Formula gaussiana usada por el pipeline:

```text
NLPD = mean(0.5 * log(2*pi*sigma^2) + (y_true - mu)^2 / (2*sigma^2))
```

donde `mu` es la media predictiva y `sigma` la desviacion estandar predictiva.

Interpretacion:

- Mide calidad probabilistica de la prediccion.
- Penaliza errores grandes.
- Penaliza estar demasiado seguro cuando falla.
- Penaliza incertidumbre excesiva si no esta justificada.
- Cuanto menor, mejor.

Uso exacto:

- Se calcula en `src/analysis/surrogate_metrics.py` solo si hay `std_pred`.
- Para GP debe estar disponible porque `src/models/gp.py` devuelve `return_std=True`.
- Para Dummy no debe interpretarse porque no devuelve incertidumbre.
- Se guarda en folds y summaries como diagnostico probabilistico.

Criterio de lectura:

- Es diagnostico, no primary.
- Si MAE es bueno pero NLPD es malo, el modelo puede predecir bien de media pero no cuantifica bien la incertidumbre.
- Si el optimizador del GP produce warnings de convergencia, los parametros del kernel y el NLPD deben leerse con cautela.

## Coverage 95

Formula:

```text
coverage_95 = mean(y_true >= mu - 1.96*sigma and y_true <= mu + 1.96*sigma)
```

Interpretacion:

- Fraccion de muestras reales que caen dentro del intervalo predictivo al 95%.
- El valor ideal es aproximadamente 0.95.
- No siempre "mas alto es mejor": coverage muy alto puede indicar intervalos demasiado anchos.
- Coverage bajo indica sobreconfianza o incertidumbre infravalorada.

Uso exacto:

- Se calcula en `src/analysis/surrogate_metrics.py` si existe `std_pred`.
- `src/models/base.py` conserva tambien el alias `coverage95`.
- `src/analysis/tuning.py` agrega `coverage_95` en macro y micro.
- `tuning_visual_report.py` lo usa en `stability_coverage95.png`.
- `audit_entomotive_real_case.py` lo usa en `gp_coverage95_active.png`.

Criterio de lectura:

- Es diagnostico de calibracion de GP.
- No se usa para seleccionar hiperparametros en el pipeline actual.
- Debe acompanar a MAE para evitar concluir que un modelo es bueno solo porque reduce error puntual.

## Macro y micro

Macro:

- Promedia metricas por fold/dieta.
- Cada dieta pesa igual, aunque tenga 2 o 3 replicas validas.
- Es la lectura preferente para comparar dietas en LODO, porque no deja que una dieta con mas replicas pese mas.

Micro:

- Agrega ponderando por numero de muestras.
- MAE micro usa la suma de errores absolutos ponderada por muestras.
- RMSE micro usa la suma de errores cuadraticos ponderada por muestras.
- Coverage micro suma conteos de muestras dentro del intervalo.

Uso recomendado:

- Para seleccion y comunicacion principal del caso real: `mae_macro_mean`.
- Para chequeo adicional: comparar macro vs micro. Si difieren mucho, revisar folds con diferente numero de replicas validas.

## Modelos y kernels activos sin TPC

Comparacion base por cada target activo:

- `Dummy`: baseline medio/mediana.
- `GP_Linear`: `DotProduct + WhiteKernel`.
- `GP_RBF_NoARD`: `RBF` escalar + `WhiteKernel`.
- `GP_Matern32_NoARD`: `Matern(nu=1.5)` escalar + `WhiteKernel`.
- `GP_Matern52_NoARD`: `Matern(nu=2.5)` escalar + `WhiteKernel`.
- `GP_Compuesto_NoARD`: composicion aditiva lineal + Matern 5/2 + ruido.

Kernel compuesto:

```text
k(x, x') = C1 * k_Lineal(x, x') + C2 * k_Matern_5/2(x, x') + k_White(x, x')
```

ARD selectivo:

- Solo se prueba ARD sobre el mejor kernel previo por target y modo de features.
- Para la configuracion actual, ignorando TPC:
  - `REDUCED_FEATURES / FCR`: `GP_RBF_ARD`.
  - `REDUCED_FEATURES / Quitina`: `GP_RBF_ARD`.
  - `REDUCED_FEATURES / Proteina`: `GP_Matern32_ARD`.
  - `FULL_FEATURES / FCR`: `GP_Matern52_ARD`.
  - `FULL_FEATURES / Quitina`: `GP_Matern52_ARD`.
  - `FULL_FEATURES / Proteina`: `GP_Matern32_ARD`.
- `GP_Compuesto_ARD` existe como plantilla en codigo, pero no forma parte del alcance activo salvo que se seleccione explicitamente en una prueba futura.

## Ranking observado de dietas sin TPC

Esto no es una metrica de ajuste del surrogate. Es un score exploratorio sobre datos observados por dieta para decidir que muestra observada merece exploracion.

En la variante sin TPC debe usar solo:

- `FCR` invertido, porque menor FCR es mejor.
- `Quitina`, porque mayor quitina es mejor.
- `Proteina`, porque mayor proteina es mejor.

Normalizacion esperada:

```text
score_fcr = (max(fcr_mean) - fcr_mean) / (max(fcr_mean) - min(fcr_mean))
score_quitina = (quitina_mean - min(quitina_mean)) / (max(quitina_mean) - min(quitina_mean))
score_proteina = (proteina_mean - min(proteina_mean)) / (max(proteina_mean) - min(proteina_mean))
exploratory_equal_weight_score = mean(score_fcr, score_quitina, score_proteina)
```

Pesos:

- FCR: 1/3.
- Quitina: 1/3.
- Proteina: 1/3.
- TPC: 0, excluido.

Uso:

- Sirve como ranking observado y trazable de dietas ya ensayadas.
- No debe mezclarse con el criterio de seleccion de hiperparametros, que sigue siendo MAE.
- No debe presentarse como optimizacion automatica de una dieta nueva.

## Como interpretar una conclusion tecnica

Una conclusion valida del caso real sin TPC debe cumplir:

- El target analizado es uno de `FCR`, `Quitina`, `Proteina`.
- La comparacion de modelos usa solo Dummy y GP.
- El modelo ganador se justifica por MAE, preferiblemente `mae_macro_mean`.
- RMSE se usa para detectar errores grandes.
- R2 se usa como lectura secundaria de ajuste global.
- NLPD y coverage_95 se usan solo para diagnosticar incertidumbre GP.
- La validacion es LODO por dieta y no mezcla replicas de una dieta entre train y test.
- Las graficas no contienen TPC como target ni modelos de arboles/ensembles.
- Las tablas de parametros GP muestran `alpha` seleccionado y `kernel_` optimizado.

## Estado verificado tras la corrida no TPC

La variante sin TPC ya esta materializada en codigo y artefactos:

- `src/configs/tuning_specs.py` itera solo `FCR`, `Quitina` y `Proteina` en `TARGET_MAP`.
- `src/configs/tuning_specs.py` ya no contiene entradas ARD para TPC en `SELECTED_ARD_BY_CASE`.
- `run_exhaustive_tuning.py` apunta a `outputs/logs/tuning/productivity_hermetia_gp_ard_kernels_no_tpc_v1`.
- `tuning_visual_report.py` apunta a `outputs/plots/comprehensive_report_gp_ard_kernels_no_tpc_v1`.
- `audit_entomotive_real_case.py` apunta a `outputs/reports/entomotive_real_case_no_tpc_v1` y `outputs/plots/entomotive_real_case_no_tpc_v1`.
- `audit_entomotive_real_case.py` recalcula el ranking observado con tres componentes: FCR invertido, Quitina y Proteina.
- `audit_entomotive_real_case.py` genera la grafica de objetivos observados sin TPC.
- No existe `src/metrics/regression.py`; las metricas se encuentran en `src/analysis/surrogate_metrics.py` y se invocan desde `src/models/base.py`.

Los artefactos existentes con sufijo `gp_ard_kernels_v1` deben tratarse como ejecucion historica con TPC. La version limpia para redaccion y revision del caso real es la de sufijo `no_tpc_v1`.
