# Validacion del caso real Entomotive

Este archivo es una auditoria tecnica reproducible del caso real. No redacta el TFG.

## Estado

- Dataset validado: `data\entomotive_datasets\productivity_hermetia_lote.csv`.
- Checks PASS: 52.
- Checks WARN: 0.
- Modelos activos considerados: Dummy, GP_Compuesto_NoARD, GP_Linear, GP_Matern32_ARD, GP_Matern32_NoARD, GP_Matern52_ARD, GP_Matern52_NoARD, GP_RBF_ARD, GP_RBF_NoARD.
- Benchmarks y pruebas LODO de benchmark quedan fuera de alcance.

## Dataset

- 33 filas, 43 columnas.
- 11 dietas, 3 replicas por dieta.
- Especie: Hermetia.
- Targets activos: FCR, QUITINA (%) y PROTEINA (%).
- TPC_larva_media queda excluido como target en esta corrida por no ser concluyente.
- FCR tiene 31 valores validos; quitina y proteina tienen 33.

## Inventario Entomotive completo

El Excel de Entomotive genera datasets para dos especies: Hermetia y Tenebrio. El pipeline de resultados auditado aqui usa Hermetia, pero Tenebrio tambien queda generado como dataset disponible.

| dataset                        |   rows |   columns | species            |   n_diets | role                                 |
|:-------------------------------|-------:|----------:|:-------------------|----------:|:-------------------------------------|
| productivity_hermetia_lote.csv |     33 |        43 | Hermetia           |        11 | dataset modelado en el pipeline real |
| productivity_tenebrio_lote.csv |     57 |        51 | Tenebrio           |        19 | dataset generado disponible          |
| productivity_all_lote.csv      |     90 |        51 | Hermetia, Tenebrio |        19 | dataset generado disponible          |
| quality_hermetia_dieta.csv     |     11 |        40 | Hermetia           |        11 | dataset generado disponible          |
| quality_tenebrio_dieta.csv     |     19 |        42 | Tenebrio           |        19 | dataset generado disponible          |
| quality_all_dieta.csv          |     30 |        42 | Hermetia, Tenebrio |        19 | dataset generado disponible          |

## Criterio de seleccion de especie modelada

La prediccion del caso real se hace sobre Hermetia. El criterio efectivo no es que Tenebrio sea invalido, sino que Hermetia constituye el bloque mas homogeneo usado por el pipeline principal: 11 dietas, 3 replicas por dieta, todas las filas en `study_block = main` y covariables completas para los dos conjuntos de features.

Tenebrio tambien esta generado, pero mezcla `main`, `coffee_orujillo` y `water_control`, incorpora tipos adicionales de dieta y presenta distinta disponibilidad de targets. Por eso debe tratarse como segundo caso potencial o subanalisis separado, no mezclarse directamente con Hermetia en los resultados actuales.

## Como se llega al dataset

1. `notebooks/02_datasets_creator.ipynb` limpia hojas del Excel Entomotive.
2. Enumera replicas con `add_replica`.
3. Une productividad, composicion larvaria, composicion de dieta y TPC de dieta usado como covariable.
4. Extrae metadatos con `parse_diet_name`.
5. Calcula ratios nutricionales con `add_ratios`.
6. Exporta `data/entomotive_datasets/productivity_hermetia_lote.csv`.

## Dietas que destacan por objetivo observado

| target       | criterion   | diet_name   | byproduct_type   |   inclusion_pct |   mean_value |
|:-------------|:------------|:------------|:-----------------|----------------:|-------------:|
| FCR          | min         | Quinoa30    | quinoa           |              30 |      1.54322 |
| QUITINA (%)  | max         | Orujo70     | orujo            |              70 |     15.3184  |
| PROTEINA (%) | max         | Orujo50     | orujo            |              50 |     32.1474  |

## Ranking exploratorio observado

Este ranking usa pesos iguales sobre FCR invertido, quitina y proteina. Es util como cribado tecnico, no como verdad biologica cerrada.

| diet_name   | byproduct_type   |   inclusion_pct |   fcr_mean |   quitina_mean |   proteina_mean |   exploratory_equal_weight_score |
|:------------|:-----------------|----------------:|-----------:|---------------:|----------------:|---------------------------------:|
| Hoja15      | hoja             |              15 |    1.60783 |        9.2981  |         31.4171 |                         0.768455 |
| Orujo50     | orujo            |              50 |    1.73324 |        9.26318 |         32.1474 |                         0.741198 |
| Quinoa30    | quinoa           |              30 |    1.54322 |        7.17372 |         29.9841 |                         0.679731 |
| Orujo70     | orujo            |              70 |    1.90748 |       15.3184  |         27.4017 |                         0.663004 |
| Control     | control          |               0 |    1.61628 |        3.91658 |         31.9362 |                         0.6272   |
| Orujo30     | orujo            |              30 |    1.85525 |        8.9083  |         30.3219 |                         0.608984 |

Con ese criterio explicito, la dieta observada mejor posicionada es `Hoja15`.

## Evaluacion del surrogate

El caso real se evalua con LODO por dieta: cada fold deja fuera una dieta completa. Esto evita fuga entre replicas de la misma dieta.

| feature_mode     | target   | model              |   n_folds |   n_samples |   mae_macro_mean |   rmse_macro_mean |   coverage_95_macro_mean |
|:-----------------|:---------|:-------------------|----------:|------------:|-----------------:|------------------:|-------------------------:|
| REDUCED_FEATURES | FCR      | Dummy              |        11 |          31 |         0.290777 |          0.316979 |               nan        |
| REDUCED_FEATURES | FCR      | GP_Linear          |        11 |          31 |         0.270227 |          0.30966  |                 0.772727 |
| REDUCED_FEATURES | FCR      | GP_RBF_NoARD       |        11 |          31 |         0.249934 |          0.281893 |                 0.545455 |
| REDUCED_FEATURES | FCR      | GP_Matern32_NoARD  |        11 |          31 |         0.252774 |          0.284973 |                 0.545455 |
| REDUCED_FEATURES | FCR      | GP_Matern52_NoARD  |        11 |          31 |         0.254439 |          0.285967 |                 0.545455 |
| REDUCED_FEATURES | FCR      | GP_Compuesto_NoARD |        11 |          31 |         0.248133 |          0.280863 |                 0.575758 |
| REDUCED_FEATURES | FCR      | GP_RBF_ARD         |        11 |          31 |         0.26714  |          0.303008 |                 0.590909 |
| REDUCED_FEATURES | Quitina  | Dummy              |        11 |          33 |         2.29274  |          2.49982  |               nan        |
| REDUCED_FEATURES | Quitina  | GP_Linear          |        11 |          33 |         2.08414  |          2.25899  |                 0.848485 |
| REDUCED_FEATURES | Quitina  | GP_RBF_NoARD       |        11 |          33 |         1.96461  |          2.06557  |                 0.727273 |
| REDUCED_FEATURES | Quitina  | GP_Matern32_NoARD  |        11 |          33 |         1.99351  |          2.09057  |                 0.848485 |
| REDUCED_FEATURES | Quitina  | GP_Matern52_NoARD  |        11 |          33 |         1.99279  |          2.09299  |                 0.818182 |
| REDUCED_FEATURES | Quitina  | GP_Compuesto_NoARD |        11 |          33 |         1.93924  |          2.084    |                 0.69697  |
| REDUCED_FEATURES | Quitina  | GP_RBF_ARD         |        11 |          33 |         2.70908  |          2.84007  |                 0.545455 |
| REDUCED_FEATURES | Proteina | Dummy              |        11 |          33 |         3.08364  |          3.21743  |               nan        |
| REDUCED_FEATURES | Proteina | GP_Linear          |        11 |          33 |         2.42637  |          2.50652  |                 0.727273 |
| REDUCED_FEATURES | Proteina | GP_RBF_NoARD       |        11 |          33 |         2.46314  |          2.69087  |                 0.787879 |
| REDUCED_FEATURES | Proteina | GP_Matern32_NoARD  |        11 |          33 |         2.4162   |          2.61941  |                 0.757576 |
| REDUCED_FEATURES | Proteina | GP_Matern52_NoARD  |        11 |          33 |         2.41633  |          2.6316   |                 0.727273 |
| REDUCED_FEATURES | Proteina | GP_Compuesto_NoARD |        11 |          33 |         2.02478  |          2.13488  |                 0.666667 |
| REDUCED_FEATURES | Proteina | GP_Matern32_ARD    |        11 |          33 |         2.53941  |          2.74337  |                 0.727273 |
| FULL_FEATURES    | FCR      | Dummy              |        11 |          31 |         0.290777 |          0.316979 |               nan        |
| FULL_FEATURES    | FCR      | GP_Linear          |        11 |          31 |         0.313034 |          0.354353 |                 0.772727 |
| FULL_FEATURES    | FCR      | GP_RBF_NoARD       |        11 |          31 |         0.24819  |          0.280532 |                 0.575758 |
| FULL_FEATURES    | FCR      | GP_Matern32_NoARD  |        11 |          31 |         0.260675 |          0.292767 |                 0.621212 |
| FULL_FEATURES    | FCR      | GP_Matern52_NoARD  |        11 |          31 |         0.252055 |          0.284105 |                 0.621212 |
| FULL_FEATURES    | FCR      | GP_Compuesto_NoARD |        11 |          31 |         0.244741 |          0.278188 |                 0.454545 |
| FULL_FEATURES    | FCR      | GP_Matern52_ARD    |        11 |          31 |         0.27072  |          0.301968 |                 0.666667 |
| FULL_FEATURES    | Quitina  | Dummy              |        11 |          33 |         2.29274  |          2.49982  |               nan        |
| FULL_FEATURES    | Quitina  | GP_Linear          |        11 |          33 |         2.24999  |          2.38777  |                 0.848485 |
| FULL_FEATURES    | Quitina  | GP_RBF_NoARD       |        11 |          33 |         2.12706  |          2.23136  |                 0.787879 |
| FULL_FEATURES    | Quitina  | GP_Matern32_NoARD  |        11 |          33 |         2.03437  |          2.13852  |                 0.818182 |
| FULL_FEATURES    | Quitina  | GP_Matern52_NoARD  |        11 |          33 |         2.01908  |          2.12503  |                 0.757576 |
| FULL_FEATURES    | Quitina  | GP_Compuesto_NoARD |        11 |          33 |         1.94046  |          2.07999  |                 0.69697  |
| FULL_FEATURES    | Quitina  | GP_Matern52_ARD    |        11 |          33 |         2.4894   |          2.61956  |                 0.818182 |
| FULL_FEATURES    | Proteina | Dummy              |        11 |          33 |         3.08364  |          3.21743  |               nan        |
| FULL_FEATURES    | Proteina | GP_Linear          |        11 |          33 |         2.61364  |          2.69396  |                 0.757576 |
| FULL_FEATURES    | Proteina | GP_RBF_NoARD       |        11 |          33 |         2.40097  |          2.60913  |                 0.727273 |
| FULL_FEATURES    | Proteina | GP_Matern32_NoARD  |        11 |          33 |         2.30956  |          2.52451  |                 0.757576 |
| FULL_FEATURES    | Proteina | GP_Matern52_NoARD  |        11 |          33 |         2.31819  |          2.53628  |                 0.787879 |
| FULL_FEATURES    | Proteina | GP_Compuesto_NoARD |        11 |          33 |         1.95435  |          2.08752  |                 0.69697  |
| FULL_FEATURES    | Proteina | GP_Matern32_ARD    |        11 |          33 |         2.64994  |          2.76749  |                 0.666667 |

## Figuras limpias generadas

- `outputs\plots\entomotive_real_case_no_tpc_v1\dataset_targets_by_diet.png`
- `outputs\plots\entomotive_real_case_no_tpc_v1\observed_equal_weight_ranking.png`
- `outputs\plots\entomotive_real_case_no_tpc_v1\active_mae_reduced_features.png`
- `outputs\plots\entomotive_real_case_no_tpc_v1\active_mae_full_features.png`
- `outputs\plots\entomotive_real_case_no_tpc_v1\gp_coverage95_active.png`

## Conclusion tecnica

El caso real esta listo para trazar el dataset Entomotive, justificar como se construye y evaluar el surrogate sobre dietas no vistas.

Tambien permite proponer dietas observadas candidatas para exploracion posterior, siempre que se declare el objetivo. Los objetivos individuales quedan recogidos en la tabla de candidatos y el cribado multiobjetivo usa solo FCR, quitina y proteina.

Lo que no queda cerrado por el codigo actual es la generacion automatica de una formulacion nueva no observada. Para eso haria falta definir un espacio de dietas validas y una regla final de adquisicion/ranking.

## Checks

| check                                              | status   | detail                                                                                                                                                                                                                                          |
|:---------------------------------------------------|:---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset_exists                                     | PASS     | C:\Users\maria\Downloads\Jose_lenovo\TFG\surrogate_models\data\entomotive_datasets\productivity_hermetia_lote.csv                                                                                                                               |
| dataset_shape                                      | PASS     | 33 rows, 43 columns                                                                                                                                                                                                                             |
| species                                            | PASS     | Hermetia                                                                                                                                                                                                                                        |
| diet_replicates                                    | PASS     | 11 diets; replicate range 3-3                                                                                                                                                                                                                   |
| target_FCR                                         | PASS     | FCR: 31 valid values                                                                                                                                                                                                                            |
| target_Quitina                                     | PASS     | QUITINA (%): 33 valid values                                                                                                                                                                                                                    |
| target_Proteina                                    | PASS     | PROTEINA (%): 33 valid values                                                                                                                                                                                                                   |
| features_REDUCED_FEATURES                          | PASS     | missing=[]; nulls={'inclusion_pct': 0, 'Proteína (%)_media': 0, 'Fibra (%)_media': 0, 'Grasa (%)_media': 0, 'TPC_dieta_media': 0}                                                                                                               |
| features_FULL_FEATURES                             | PASS     | missing=[]; nulls={'inclusion_pct': 0, 'Proteína (%)_media': 0, 'Grasa (%)_media': 0, 'Fibra (%)_media': 0, 'Cenizas (%)_media': 0, 'Carbohidratos (%)_media': 0, 'ratio_P_C': 0, 'ratio_P_F': 0, 'ratio_Fibra_Grasa': 0, 'TPC_dieta_media': 0} |
| byproduct_type                                     | PASS     | {'orujo': 12, 'hoja': 9, 'quinoa': 9, 'control': 3}                                                                                                                                                                                             |
| folds_REDUCED_FEATURES_FCR_Dummy                   | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_FCR_GP_Linear               | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_FCR_GP_RBF_NoARD            | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_FCR_GP_Matern32_NoARD       | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_FCR_GP_Matern52_NoARD       | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_FCR_GP_Compuesto_NoARD      | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_FCR_GP_RBF_ARD              | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Quitina_Dummy               | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Quitina_GP_Linear           | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Quitina_GP_RBF_NoARD        | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Quitina_GP_Matern32_NoARD   | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Quitina_GP_Matern52_NoARD   | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Quitina_GP_Compuesto_NoARD  | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Quitina_GP_RBF_ARD          | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Proteina_Dummy              | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Proteina_GP_Linear          | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Proteina_GP_RBF_NoARD       | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Proteina_GP_Matern32_NoARD  | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Proteina_GP_Matern52_NoARD  | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Proteina_GP_Compuesto_NoARD | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_REDUCED_FEATURES_Proteina_GP_Matern32_ARD    | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_FCR_Dummy                      | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_FCR_GP_Linear                  | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_FCR_GP_RBF_NoARD               | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_FCR_GP_Matern32_NoARD          | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_FCR_GP_Matern52_NoARD          | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_FCR_GP_Compuesto_NoARD         | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_FCR_GP_Matern52_ARD            | PASS     | 11 folds, 31 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Quitina_Dummy                  | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Quitina_GP_Linear              | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Quitina_GP_RBF_NoARD           | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Quitina_GP_Matern32_NoARD      | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Quitina_GP_Matern52_NoARD      | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Quitina_GP_Compuesto_NoARD     | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Quitina_GP_Matern52_ARD        | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Proteina_Dummy                 | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Proteina_GP_Linear             | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Proteina_GP_RBF_NoARD          | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Proteina_GP_Matern32_NoARD     | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Proteina_GP_Matern52_NoARD     | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Proteina_GP_Compuesto_NoARD    | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
| folds_FULL_FEATURES_Proteina_GP_Matern32_ARD       | PASS     | 11 folds, 33 samples                                                                                                                                                                                                                            |
