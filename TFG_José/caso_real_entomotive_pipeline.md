   # Trazabilidad tecnica del caso real Entomotive

   Este documento no redacta el TFG. Su objetivo es dejar trazado el caso real:

   - como se genera el dataset de dietas de Hermetia;
   - que codigo lo usa;
   - que graficas existen;
   - si el pipeline permite defender una frase del tipo: "esta es la muestra que merece ser explorada".

Alcance: se ignoran benchmarks y pruebas LODO de benchmark. Para el caso real, la lectura vigente del codigo se centra en `Dummy`, kernels GP sin ARD (`GP_Linear`, `GP_RBF_NoARD`, `GP_Matern32_NoARD`, `GP_Matern52_NoARD`, `GP_Compuesto_NoARD`) y una variante ARD selectiva para el kernel que fue mejor en la comparacion previa de cada target/modo. La corrida vigente excluye `TPC_larva_media` como target porque no fue concluyente; se mantienen `FCR`, `Quitina` y `Proteina`.

## 1. Fuente y datasets Entomotive

Fuente bruta:

- `notebooks/Datos finales para analisis IA _proyecto ENTOMOTIVE_.xlsx`

   Notebook que genera los CSV:

   - `notebooks/02_datasets_creator.ipynb`

El Excel genera datasets procesados para dos especies:

- `Hermetia`
- `Tenebrio`

Datasets procesados disponibles:

| Dataset | Especie | Filas | Columnas | Dietas | Papel |
|---|---|---:|---:|---:|---|
| `productivity_hermetia_lote.csv` | Hermetia | 33 | 43 | 11 | dataset modelado en el pipeline real |
| `productivity_tenebrio_lote.csv` | Tenebrio | 57 | 51 | 19 | dataset generado disponible |
| `productivity_all_lote.csv` | Hermetia + Tenebrio | 90 | 51 | 19 | union de productividad |
| `quality_hermetia_dieta.csv` | Hermetia | 11 | 40 | 11 | calidad agregada por dieta |
| `quality_tenebrio_dieta.csv` | Tenebrio | 19 | 42 | 19 | calidad agregada por dieta |
| `quality_all_dieta.csv` | Hermetia + Tenebrio | 30 | 42 | 19 | union de calidad |

Dataset final usado por los resultados del caso real:

- `data/entomotive_datasets/productivity_hermetia_lote.csv`

Estado verificado del CSV modelado:

   - 33 filas y 43 columnas.
   - 11 dietas (`diet_name`).
   - 3 replicas por dieta.
   - Especie unica: `Hermetia`.
   - Subproductos: `control`, `hoja`, `orujo`, `quinoa`.
   - Targets activos usados:
   - `FCR`: 31 valores validos; faltan 2.
   - `QUITINA (%)`: 33 valores validos.
   - `PROTEINA (%)`: 33 valores validos.
   - `TPC_larva_media`: 33 valores validos disponibles en el CSV, pero excluido como target en la corrida vigente.

   Los dos valores ausentes de `FCR` estan en dietas que conservan replicas validas:

   - `Hoja50`
   - `Orujo90`

## 2. Como se generan Hermetia y Tenebrio

   La construccion esta en `notebooks/02_datasets_creator.ipynb`.

   ### 2.1 Limpieza comun

   El notebook define funciones auxiliares para limpiar y estructurar las hojas del Excel:

   - `make_unique_columns(df)`: evita nombres de columna duplicados.
   - `norm_treat(x)`: normaliza el tratamiento.
   - `clean_mean_sd_sheet_from_header12(sheet_name)`: limpia hojas de composicion de dieta con medias y desviaciones.
   - `clean_productivos(sheet_name)`: limpia parametros productivos y normaliza `FCR`.
   - `clean_larvas(sheet_name)`: limpia composicion de larvas.
   - `clean_tpc(sheet_name)`: limpia TPC y lo deja como:
   - `TPC_dieta_media`
   - `TPC_dieta_sd`
   - `TPC_larva_media`
   - `TPC_larva_sd`
   - `add_replica(df, by_cols)`: enumera replicas con `groupby(...).cumcount() + 1`.
   - `parse_diet_name(dieta)`: extrae metadatos de dieta.
   - `add_ratios(df)`: calcula ratios nutricionales.

   Referencias en notebook:

   - Definicion de `add_replica`: `notebooks/02_datasets_creator.ipynb`, entorno de la linea 108.
   - Definicion de `parse_diet_name`: entorno de la linea 115.
   - Definicion de `add_ratios`: entorno de la linea 163.

   ### 2.2 Metadatos de dieta

   `parse_diet_name` convierte el nombre textual de la dieta en variables explicitas:

   - `diet_name`
   - `byproduct_type`
   - `inclusion_pct`
   - `water_condition`
   - `processing`
   - `study_block`

   Esto es importante porque el modelo no trabaja solo con el nombre de la dieta, sino con composicion, inclusion y tipo de subproducto.

### 2.3 Dataset de productividad Hermetia

   Bloque Hermetia del notebook, entorno de las lineas 1614-1621:

   1. Se enumeran replicas por dieta y tratamiento:
      - `prod_H_l = add_replica(prod_H, ["Dieta","Tratamiento"])`
      - `larv_H_l = add_replica(larv_H, ["Dieta","Tratamiento"])`

   2. Se etiqueta la especie:
      - `prod_H_l["species"] = "Hermetia"`
      - `larv_H_l["species"] = "Hermetia"`

   3. Se unen parametros productivos y composicion de larvas por:
      - `Dieta`
      - `Tratamiento`
      - `Replica`
      - `species`

   4. Se anade composicion de la dieta con merge por:
      - `Dieta`
      - `Tratamiento`

   5. Se anade TPC con merge por:
      - `Dieta`
      - `Tratamiento`

   6. Se generan metadatos estructurados con `parse_diet_name`.

   7. Se reordenan columnas para dejar una tabla consistente.

   Exportacion final, entorno de las lineas 3577-3586:

   - `save_csv(prod_H_ds, "productivity_hermetia_lote")`

Por tanto, cada fila final representa un lote/replica de una dieta de Hermetia, enriquecido con:

   - resultados productivos;
   - composicion de larva;
   - composicion nutricional de la dieta;
- TPC de dieta y larva;
- metadatos de dieta.

### 2.4 Dataset de productividad Tenebrio

Tenebrio se construye en el mismo notebook y no debe desaparecer de la explicacion del origen de datos.

Bloque Tenebrio del notebook, entorno de las lineas 1624-1649:

1. Se enumeran replicas:
   - `prod_T_l = add_replica(prod_T, ["Dieta"])`
   - `larv_T_l = add_replica(larv_T, ["Dieta"])`

2. Se etiqueta la especie:
   - `prod_T_l["species"] = "Tenebrio"`
   - `larv_T_l["species"] = "Tenebrio"`

3. Se unen productividad y composicion larvaria por:
   - `Dieta`
   - `Replica`
   - `species`

4. Se crean columnas de composicion:
   - `Dieta_comp`
   - `Trat_comp`
   - `Tratamiento`

5. Se anade composicion de dieta con merge frente a `diet_T`.

6. Se anade TPC con merge frente a `tpc_T`.

7. Se generan metadatos con `parse_diet_name`.

8. Se exporta:
   - `save_csv(prod_T_ds, "productivity_tenebrio_lote")`

Estado del CSV generado:

- `data/entomotive_datasets/productivity_tenebrio_lote.csv`
- 57 filas y 51 columnas.
- 19 dietas.
- Especie unica: `Tenebrio`.

La diferencia importante es que `run_exhaustive_tuning.py` y `tuning_visual_report.py` cargan explicitamente `productivity_hermetia_lote.csv`. Por tanto, Tenebrio esta trazado como dataset Entomotive generado, pero no entra en los resultados modelados actuales del caso real.

### 2.5 Criterio de seleccion de la especie modelada

La prediccion del caso real se hace solo sobre `Hermetia`, no sobre `Tenebrio`.

El criterio efectivo en el repositorio es de alcance experimental y homogeneidad del caso modelado:

1. El pipeline principal selecciona explicitamente `productivity_hermetia_lote.csv`:
   - `run_exhaustive_tuning.py` carga `ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"`.
   - `tuning_visual_report.py` usa el mismo CSV como `DATA_FILE`.

2. Hermetia forma un bloque experimental homogeneo:
   - 33 filas.
   - 11 dietas.
   - 3 replicas por dieta.
   - `study_block = main` en todas las filas.
   - targets activos casi completos: `FCR` con 31 valores y quitina/proteina con 33; `TPC_larva_media` existe pero queda excluido como target en la corrida vigente.

3. Tenebrio esta generado, pero mezcla mas bloques experimentales:
   - 57 filas.
   - 19 dietas.
   - `study_block = main`, `coffee_orujillo` y `water_control`.
   - incluye tipos adicionales como `orujillo` y `cafe`.
   - no todos los targets tienen el mismo numero de muestras validas: `FCR` tiene 57, `PROTEINA (%)` 56, pero `TPC_larva_media` y `QUITINA (%)` tienen 33.

Por tanto, la seleccion de Hermetia no significa que Tenebrio sea invalido. Significa que los resultados actuales del surrogate se limitan a un caso real mas homogeneo y directamente validable con LODO por dieta. Tenebrio queda como segundo caso potencial, que requeriria definir antes si se filtra solo el bloque `main` o si se modelan tambien los bloques `coffee_orujillo` y `water_control`.

   ## 3. Variables que entran al modelo

   Configuracion:

   - `src/configs/tuning_specs.py`

   Targets:

   ```python
   TARGET_MAP = {
      "FCR": "FCR",
      "Quitina": "QUITINA (%)",
      "Proteina": "PROTEINA (%)",
   }
   ```

   Features reducidas:

   ```python
   FEATURE_COLS_REDUCED = [
      "inclusion_pct",
      "Proteína (%)_media",
      "Fibra (%)_media",
      "Grasa (%)_media",
      "TPC_dieta_media",
   ]
   ```

   Features completas:

   ```python
   FEATURE_COLS_FULL = [
      "inclusion_pct",
      "Proteína (%)_media",
      "Grasa (%)_media",
      "Fibra (%)_media",
      "Cenizas (%)_media",
      "Carbohidratos (%)_media",
      "ratio_P_C",
      "ratio_P_F",
      "ratio_Fibra_Grasa",
      "TPC_dieta_media",
   ]
   ```

   Ademas, el pipeline anade dummies de `byproduct_type`.

   Resultado:

   - `REDUCED_FEATURES`: 5 variables numericas + 4 dummies = 9 features.
   - `FULL_FEATURES`: 10 variables numericas + 4 dummies = 14 features.

   En el CSV actual, las features usadas no tienen nulos. Por tanto, el `fillna(median)` del pipeline no cambia los datos del caso real.

   ## 4. Como se construyen X, y y grupos

   Archivo:

   - `run_exhaustive_tuning.py`

   Funcion:

   - `build_X_y_groups(df, target_col, feature_cols)`, desde la linea 32.

   Pasos:

   1. Filtra filas sin target:
      - `data = data.loc[~data[target_col].isna()]`

   2. Define grupos por dieta:
      - `groups = data["diet_name"].astype(str).to_numpy()`

   3. Define el vector respuesta:
      - `y = data[target_col].astype(float).to_numpy()`

   4. Crea dummies de tipo de subproducto:
      - `pd.get_dummies(data["byproduct_type"], prefix="byproduct")`

   5. Junta variables numericas y dummies.

   6. Convierte a matriz numerica `X`.

   Esto es correcto para el objetivo metodologico del caso real: evaluar generalizacion a dietas no vistas, no a replicas aleatorias.

   ## 5. Entrenamiento y validacion del caso real

   El dataset se carga en:

   - `run_exhaustive_tuning.py`, linea 117:
   - `ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"`

   Los resultados de la corrida vigente se guardan bajo:

   - `outputs/logs/tuning/TFG_MAIN_real_case_hermetia_no_tpc_tuning`

   Validacion:

   - `src/analysis/tuning.py`
   - `nested_lodo_tuning(...)`, desde la linea 133.

   Estructura:

   1. Outer LODO:
      - cada fold deja fuera una dieta completa.
      - con 11 dietas, hay 11 folds.

   2. Inner LODO:
      - se usa solo con las dietas de entrenamiento del fold externo.
      - selecciona hiperparametros minimizando MAE.

   3. Entrenamiento final del fold:
      - se entrena con todas las dietas salvo la retenida.
      - se predice sobre la dieta retenida.

   4. Metricas:
      - `mae`
      - `rmse`
      - `r2`
      - `max_error`
      - para los GP tambien incertidumbre: `nlpd`, `coverage_95`, anchura de intervalo y `sharpness`.

   Esta definicion es metodologicamente correcta para comprobar si el modelo extrapola a dietas no vistas dentro del espacio experimental observado.

   ## 6. Modelos activos para el caso real

   En el estado actual de `src/configs/tuning_specs.py`, los modelos activos son:

   ```python
   MODELS = {
       "Dummy": DummySurrogateRegressor(),
      "GP_Linear": GPSurrogateRegressor(),
      "GP_RBF_NoARD": GPSurrogateRegressor(),
      "GP_Matern32_NoARD": GPSurrogateRegressor(),
      "GP_Matern52_NoARD": GPSurrogateRegressor(),
      "GP_Compuesto_NoARD": GPSurrogateRegressor(),
      # ARD selectivo segun target y modo de features
   }
   ```

   Lectura:

   - `Dummy` es baseline.
   - `GP_Linear` es el GP con kernel lineal (`DotProduct + WhiteKernel`).
   - `GP_RBF_NoARD` es el GP con kernel radial isotropico.
   - `GP_Matern32_NoARD` es el GP con kernel Matern nu=3/2 isotropico.
   - `GP_Matern52_NoARD` es el GP con kernel Matern nu=5/2 isotropico.
   - `GP_Compuesto_NoARD` es el GP con kernel aditivo:
     \(k(x,x') = C_1 k_{Lineal}(x,x') + C_2 k_{Matern5/2}(x,x') + k_{White}(x,x')\).
   - La version ARD se prueba solo para el mejor kernel previo de cada target/modo, por ejemplo `GP_RBF_ARD` si el mejor NoARD previo fue RBF.
   - No se consideran otros modelos en la lectura vigente del caso real.

   ## 7. Graficas disponibles

   Script:

   - `tuning_visual_report.py`

   Carpeta de salida:

   - `outputs/plots/TFG_MAIN_real_case_gp_report_no_tpc`

   Hay graficas para:

   - `fcr`
   - `quitina`
   - `proteina`

   Para cada target y modo de features existen graficas de este tipo:

   - `parity_plot.png`: observado vs predicho por validacion cruzada.
   - `gp_uncertainty_deep_dive_<modelo>.png`: error, incertidumbre e intervalos de cada GP.
   - `stability_mae.png`: distribucion del MAE por fold.
   - `stability_rmse.png`: distribucion del RMSE por fold.
   - `stability_coverage95.png`: estabilidad de cobertura de los GP.
   - `response_profile_<feature>.png`: perfil de respuesta del modelo frente a una variable.
   - `gp_fitted_parameters.png`: tabla grafica con `alpha` seleccionado y `kernel_` optimizado de cada GP.

   Tambien hay comparativas:

   - `comparison_full_vs_reduced_<target>.png`

   Uso recomendado:

   - Para justificar capacidad predictiva: `parity_plot.png` y `stability_mae.png`.
   - Para justificar fiabilidad/incertidumbre: `gp_uncertainty_deep_dive.png` y `stability_coverage95.png`.
   - Para discutir que variables parecen mover la prediccion: `response_profile_*.png`.
   - Para decidir entre set reducido y completo: `comparison_full_vs_reduced_*.png`.

   Precaucion:

   - Si alguna figura antigua muestra arboles u otros modelos distintos de `Dummy` y los `GP_*` definidos arriba, debe leerse como artefacto historico o excluirse de la memoria.

   ## 8. Dietas observadas que destacan por objetivo

   Tabla calculada directamente desde `productivity_hermetia_lote.csv`, agregando por dieta.

   | Objetivo | Criterio | Dieta que destaca | Media |
   |---|---|---:|---:|
   | FCR | menor es mejor | Quinoa30 | 1.543216 |
   | Quitina | mayor es mejor | Orujo70 | 15.318378 |
   | Proteina | mayor es mejor | Orujo50 | 32.147393 |

   Interpretacion:

   - Si el objetivo principal es eficiencia productiva medida por FCR, la dieta observada que destaca es `Quinoa30`.
   - Si el objetivo principal es quitina, destaca `Orujo70`.
   - Si el objetivo principal es proteina, destaca `Orujo50`.

   No hay una unica "mejor muestra" sin definir antes el objetivo biologico/productivo o una funcion multiobjetivo.

   ## 9. Puede el codigo sostener "esta es la muestra que merece ser explorada"?

   Respuesta corta: parcialmente, pero con una condicion.

   El codigo esta correctamente definido para decir:

   > "Con los datos disponibles, estas dietas observadas son candidatas razonables para exploracion posterior bajo un objetivo definido."

   El codigo no esta todavia definido para decir, de forma automatica y cerrada:

   > "Esta nueva formulacion no observada es la muestra que debe explorarse."

   Motivo:

   - El pipeline real valida capacidad predictiva con LODO por dieta.
   - Los GP aportan prediccion e incertidumbre, diferenciando explicitamente el kernel usado.
   - Las graficas permiten ver ajuste, error, estabilidad e incertidumbre.
   - Pero no hay una etapa final conectada que:
   - defina una funcion objetivo multiobjetivo;
   - genere solo dietas fisicamente validas;
   - rankee candidatos nuevos;
   - devuelva una muestra final con trazabilidad experimental.

   Hay codigo auxiliar que se aproxima a esta idea, pero no cierra el caso:

   - `src/models/base.py`, `rank_candidates(...)`: existe como ranking generico de candidatos, pero no esta conectado al pipeline real y tiene un TODO sobre FCR, porque FCR debe minimizarse.
   - `src/analysis/plotting.py`, `suggest_next_point_gp_ucb(...)`: propone un punto por UCB, pero muestrea el rango numerico de features y no garantiza que el resultado sea una dieta real o formulable.
   - `notebooks/00_data_exploration.ipynb`, `ranking_table(...)`: rankea dietas observadas por target, pero es exploratorio y no forma una recomendacion GP final.

   Conclusion tecnica:

   - Para identificar una dieta observada prometedora, el caso real esta bien trazado.
   - Para proponer una nueva muestra experimental, falta una capa explicita de recomendacion.

   ## 10. Definicion minima para cerrar la recomendacion

   Para que la frase "esta es la muestra que merece ser explorada" quede tecnicamente cerrada, haria falta anadir una decision final al pipeline del caso real:

   1. Definir objetivo:
      - minimizar `FCR`;
      - maximizar `QUITINA (%)`;
      - maximizar `PROTEINA (%)`;
      - o una funcion multiobjetivo ponderada.

   2. Definir espacio candidato:
      - solo dietas ya observadas;
      - o nuevas combinaciones validas de subproducto/inclusion/composicion.

   3. Definir criterio de exploracion:
      - mejor media predicha;
      - media penalizada por incertidumbre;
      - UCB/LCB segun objetivo;
      - o Pareto si se usa multiobjetivo.

   4. Generar una tabla final:
      - candidato;
      - features;
      - prediccion GP, indicando la familia de kernel;
      - incertidumbre;
      - evidencia observada disponible;
      - motivo por el que merece exploracion.

   Con esa capa, el pipeline pasaria de "validacion del surrogate sobre dietas no vistas" a "recomendacion trazable de una muestra para exploracion experimental".
